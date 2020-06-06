#-*- coding: utf-8 -*-
'''
 
:author: YuanZhou

description: transform incoming list of operators into cuda code
 
'''
import os
import time

import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import qutip as qt
from jinja2 import Template
from pycuda.compiler import SourceModule

base = os.path.dirname(os.path.abspath(__file__))
def load_file(file_name):
    with open(os.path.join(base,file_name), 'r') as content:
        return content.read()

class send_into_cuda():
    def __init__(self, hamiltonian, rho, tlist, c_ops, c_ops_dag, e_ops, params):
    #prepare for cuda: inject code into template.
        self.hamiltonian    = hamiltonian
        self.rho            = rho
        self.tlist          = tlist
        self.c_ops          = c_ops
        self.c_ops_dag      = c_ops_dag
        self.e_ops          = e_ops
        self.params         = params
        self.expect         = []
        self.length         = tells_length(tlist)

        # L_max = pycuda.tools.DeviceData().max_threads
        L_max = 256
        print("maximal threads per block: {}\n".format(L_max))
        self.L = len(params[0][1])*len(params[1][1])

        if self.L <= L_max:
            Bx = self.L
            Gx = 1
        else:
            Bx = L_max
            Gx = np.floor(self.L/L_max) + 1
        self.block = (Bx, 1, 1) 
        self.grid = (np.int(Gx), 1)

        dim = rho.shape[0]
        print("cuda_solver: rendering...\n")

        kernel_file = load_file("mesolve_kernel.c")
        kernel = Template(kernel_file).render(**
        {
            'dim': dim,
            'mat_size': dim**2,
            'h_args': construct_args(self.hamiltonian, "const dcmplx *", "h"),
            'c_ops': construct_args(self.c_ops, "const dcmplx *", "c_ops"),
            'c_ops_dag': construct_args(self.c_ops, "const dcmplx *", "c_ops_dag"),
            'e_ops': construct_args(self.e_ops, "const dcmplx *", "e_ops"),
            'e_args': construct_args(self.e_ops, "dcmplx *", "expect"),
            'moments': self.length,
            'thread_length': self.L,
            'ranged_param_len': len(params[1][1]),

            'dim_convert': convert_dim(self.params),
            'effect_of_Hamiltonian_k1': construct_H_code(hamiltonian, 'rho', 'k1'),
            'effect_of_Hamiltonian_k2': construct_H_code(hamiltonian, 'temp', 'k2'),
            'effect_of_Hamiltonian_k3': construct_H_code(hamiltonian, 'temp', 'k3'),

            'effect_of_Lindblad_k1': construct_L_code(c_ops, 'rho', 'k1'),
            'effect_of_Lindblad_k2': construct_L_code(c_ops, 'temp', 'k2'),
            'effect_of_Lindblad_k3': construct_L_code(c_ops, 'temp', 'k3'),
            'expects': construct_expect_code(e_ops)
        }
        )
        # print(kernel)
        print("cuda_solver: compiling...\n")
        program = SourceModule(kernel)
        self.mesolve = program.get_function("mesolve")

    def solve(self):
    #send matrixes into GPU and start calculation.
        f, t = drv.mem_get_info()
        print("video-memory usage: {}\n".format(1.0-f/float(t)))
        H = []
        for h, factor in self.hamiltonian:
            H.append(drv.In(h.astype(np.complex128)))

        rho = np.array(self.rho , dtype = np.complex128)
        rho = drv.In(rho)

        step_list = []
        if isinstance(self.tlist[0], np.ndarray):
            for t in self.tlist:
                step_list+=([(t[2]-t[1])])
        elif isinstance(self.tlist[1], float):
            step_list = self.tlist[1]*np.ones(len(self.params[1][1]))
        step_list = drv.In(np.array(step_list, dtype = np.float64))

        c_ops = []
        for c, factor in self.c_ops:
            c_ops.append(drv.In(c.astype(np.complex128)))

        c_ops_dag = []
        for c, factor in self.c_ops_dag:
            c_ops_dag.append(drv.In(c.astype(np.complex128)))

        e_ops = []
        for e, factor in self.e_ops:
            e_ops.append(drv.In(e.astype(np.complex128)))
        
        expect = []     #steady-state result, every parameters gives one expectation at last
        for e, factor in self.e_ops:
            expect.append(drv.InOut(np.ones(self.L).astype(np.complex128)))

        params = []
        for a in self.params[0][1]:
            for b in self.params[1][1]:
                params+=(a, b) 
        params= drv.In(np.array(params, dtype = np.float64))

        arguments = H + [rho] + [step_list] + c_ops + c_ops_dag + e_ops + expect + [params]
        # print(*arguments)

        print("cuda_solver: solving...\n")
        start = time.time()
        self.mesolve(*arguments, block = self.block, grid = self.grid)

        del rho
        del c_ops
        del c_ops_dag

        # np.savetxt('expect.txt', expect[0].array)
        for e in expect:
            self.expect.append(np.array(e.array.reshape(len(self.params[0][1]), len(self.params[1][1]))))
            del e

        print('cuda_solver: {} s\n'.format(time.time()-start))

def construct_args(obj, data_type, prefix):
#construce arguments in cuda code
    args = []
    if isinstance(obj, list):
        for ind, ops in enumerate(obj):
            if isinstance(ops[0], str):
                args.append('{}{}'.format(data_type, ops[0]))
            elif isinstance(ops[0], np.ndarray):
                args.append('{}{}_{}'.format(data_type, prefix, ind))
    elif isinstance(obj, np.ndarray):
        args.append('{}{}_{}'.format(data_type, prefix, 0))
    return ', '.join(args)

def convert_dim(obj):
#convert the two-dimension parameter space [a, b] into a one-dimension [a*b, 1]
    code = []
    for ind, p in enumerate(obj):
        if not isinstance(p[1], np.ndarray):
            raise ValueError('Unexpected params format {}'.format(type(p[1])))
        code.append('double {} = params[{}*offset+{}];'.format(p[0], len(obj), ind))
    return code

def construct_H_code(obj, ind1, ind2):
#construct the code about Hamiltonian
    code = []
    for i, ops in enumerate(obj):
        code+=['apply_Hamiltonian(h_{}, {}, {}, {});'.format(i, ind1, ind2, ops[1])]
    return code

def construct_L_code(obj, ind1, ind2):
#construct the code about Lindbladian
    code = []
    for i, ops in enumerate(obj):
        code += ['apply_Lindblad(c_ops_{}, c_ops_dag_{}, {}, {}, {});'.format(i, i, ind1, ind2, ops[1])]
    return code

def tells_length(tlist):
#returns the length of time list. Since the time list could be one dimensional or two dimensional one
#things are different in two cases
    if isinstance(tlist[0], np.ndarray):
        return len(tlist[0])
    elif isinstance(tlist, np.ndarray):
        return len(tlist)
    else:
        raise TypeError("Time list illegal")

def construct_expect_code(obj):
#contruct the code about expectation operator
    code = []
    for i, ops in enumerate(obj):
        code += ['expect_{}[offset] = matrix_trace(rho, e_ops_{});'.format(i, i)]
    return code
