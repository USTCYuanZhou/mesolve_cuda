#-*- coding: utf-8 -*-
'''
 
:author: YuanZhou

description: check if the incoming args were q.objs, and list them like [(matrix, factor),...]
 
'''
import qutip as qt
import numpy as np
from cuda_solver import send_into_cuda
class cuda_result(send_into_cuda):
    def __init__(self, hamiltonian, rho, tlist, c_ops, e_ops, params, tlength):
        send_into_cuda.__init__( #initial send_into_cuda, the point is get the kernel compiled
            self,
            self.table_ops(hamiltonian),
            self.list_ops(rho),
            tlist,
            self.table_ops(c_ops),
            self.table_ops(c_ops, dagger = True),  #!! construct dag !!
            self.table_ops(e_ops),
            self.list_params(params),
            tlength
        )

    def list_ops(self, obj, dagger = False):
        if isinstance(obj, list):
            ops_list = []
            for ops in obj:
                ops_list.append(self.list_ops(ops, dagger))
            return ops_list
        elif isinstance(obj, qt.Qobj):
            if dagger:
                obj = obj.dag()
            return obj.full()
        elif isinstance(obj, str):
            return obj
        else:
            raise RuntimeError("Unsupported type {}".format(type(obj)))

    def table_ops(self, obj, dagger = False):
        ops_table = []
        ops_list = self.list_ops(obj, dagger)
        if isinstance(obj, qt.Qobj):
            if dagger:
                obj = obj.dag()
            ops_table.append([obj.full(), 1.0])
            return ops_table
        for ops in ops_list:
            if isinstance(ops, list):
                if (isinstance(ops[0], np.ndarray) and isinstance(ops[1], str)):
                    ops_table.append(ops)
            else:
                ops_table.append([ops, 1.0])
        return ops_table
        
    def list_params(self, obj):
        params=[]
        vars = 0
        if isinstance(obj[0], str):
            obj =[obj]
        for k, v in obj:
            if isinstance(v, (list, np.ndarray)):
                params.append([k, np.array(v)])
                vars = vars + 1
            else:
                params.append([k, v])
        if vars > 2:
            raise ValueError("Varieties more than 2 is not supported up to now!")
        return params
    
def mesolve(hamiltonian, rho, tlist, c_ops, e_ops, params = [['no', [0]], ['ne', [0]]], tlength = 256):
    def verify_qobj(obj, allow_string = False):
        if isinstance(obj, list):
            for ops in obj:
                verify_qobj(ops, allow_string)
        elif not ((isinstance(obj, qt.Qobj)) or (allow_string) and isinstance(obj, str)):
            raise TypeError("Expected operator or string but got {}".format(type(obj)))
    
    verify_qobj(hamiltonian, allow_string = True)
    verify_qobj(c_ops, allow_string = True)
    verify_qobj(rho)
    verify_qobj(e_ops)

    result = cuda_result(hamiltonian, rho, tlist, c_ops, e_ops, params, tlength)
    result.solve()
    return result
    