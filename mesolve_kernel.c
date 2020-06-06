#include <pycuda-complex.hpp>
#include <stdio.h>
#include <math.h>

typedef pycuda::complex<double> dcmplx;
typedef unsigned short uint;

__device__
void clear_matrix(dcmplx *A){
    for(uint row = 0; row < {{dim}}; row++){
        for(uint col = 0; col < {{dim}}; col++){
            A[row*{{dim}} + col] = 0.0;
        }
    }
}

__device__
dcmplx element_of(
    const dcmplx *A,
    const dcmplx *B,
    uint row,
    uint col
){
    dcmplx element =0.0;
    for(uint i=0; i<{{dim}}; i++){
        element += A[row*{{dim}} + i]*B[i*{{dim}} + col];
    }
    return element;
}

__device__ //R = A + B + C
void add_ABC(
    const dcmplx *A,
    const dcmplx *B,
    const dcmplx *C,
    dcmplx *result,
    dcmplx factor_a,
    dcmplx factor_b,
    dcmplx factor_c
){
    for(uint row = 0; row < {{dim}}; row++){
        for(uint col = 0; col < {{dim}}; col++){
            result[row*{{dim}} + col] = factor_a*A[row*{{dim}} + col] + factor_b*B[row*{{dim}} + col] + factor_c*C[row*{{dim}} + col];
        }
    }
}

__device__ //R += A + B + C
void sum_ABC(
    const dcmplx *A,
    const dcmplx *B,
    const dcmplx *C,
    dcmplx *result,
    dcmplx factor_a,
    dcmplx factor_b,
    dcmplx factor_c
){
    for(uint row = 0; row < {{dim}}; row++){
        for(uint col = 0; col < {{dim}}; col++){
            result[row*{{dim}} + col] += factor_a*A[row*{{dim}} + col] + factor_b*B[row*{{dim}} + col] + factor_c*C[row*{{dim}} + col];
        }
    }
}

__device__ //R = A + B
void add_AB(
    const dcmplx *A,
    const dcmplx *B,
    dcmplx *result,
    dcmplx factor_a,
    dcmplx factor_b
){
    for(uint row = 0; row < {{dim}}; row++){
        for(uint col = 0; col < {{dim}}; col++){
            result[row*{{dim}} + col] = factor_a*A[row*{{dim}} + col] + factor_b*B[row*{{dim}} + col];
        }
    }
}

__device__
void mult_AB( // R += A*B
    const dcmplx *A,
    const dcmplx *B,
    dcmplx *result,
    dcmplx factor
){
    for(uint row=0; row<{{dim}}; row++){
        for(uint col=0; col<{{dim}}; col++){
            result[row*{{dim}} + col] += factor*element_of(A, B, row, col);
        }
    }
}

__device__
dcmplx matrix_trace(
    const dcmplx *A,
    const dcmplx *B
){
    dcmplx result = 0.0;
    for(uint row=0; row<{{dim}}; row++){
        result += element_of(A, B, row, row);
    }
    return result;
}

__device__
void apply_Hamiltonian(
    const dcmplx *H,
    const dcmplx *rho,
    dcmplx *result,
    dcmplx factor
){
    dcmplx imag(0,1);
    mult_AB(H, rho, result, -1.0*imag*factor);
    mult_AB(rho, H, result, +1.0*imag*factor);
}

__device__
void apply_Lindblad( // apply lindbladians to an exsisting rho
    const dcmplx *c_ops,
    const dcmplx *c_ops_dag,
    const dcmplx *rho,
    dcmplx *result,
    double factor
){

    dcmplx saving_memory[{{mat_size}}]; //local_memory

    clear_matrix(saving_memory);
    mult_AB(c_ops, rho, saving_memory, factor);
    mult_AB(saving_memory, c_ops_dag, result, 1.0);

    clear_matrix(saving_memory);
    mult_AB(c_ops_dag, c_ops, saving_memory, -0.5);
    mult_AB(saving_memory, rho, result, factor);

    clear_matrix(saving_memory);
    mult_AB(rho, c_ops_dag, saving_memory, -0.5);
    mult_AB(saving_memory, c_ops, result, factor);

}

__device__
double trapezoid(
    const double t,
    const double t0, 
    const double t1, 
    const double t2, 
    const double t3,
    const double h
){
    double result = 0.0;
    if(t <= t0||t >= t3){
        result = 0.0;
        return result;
    }
    else if(t > t0 && t <= t1){
        result = (h * (t - t0)/(t1 - t0));
        return result;
    }
    else if(t > t1 && t <= t2){
        result = h;
        return result;
    }
    else{
        result = (h * (t3 - t)/(t3 - t2));
        return result;
    }
}

__device__ 
int square(
    const double t,
    const double t0,
    const double t1
){
    int result = 0;
    if(t > t0 && t<=t1){
        result = 1;
    }
    return result;
}

__device__ 
int Heaviside(
    const double threshold,
    const double value
){
    int result = 0;
    if(value > threshold){
        result = 1;
    }
    return result;
}


__global__
void mesolve(                                   //apply a numerical solution, 3 - Runge-Kutta
    {{h_args}},
    const dcmplx *rho0,
    const double *step_list,                    //every range params could have a time step_size
    {{c_ops}},
    {{c_ops_dag}},
    {{e_ops}},
    {{e_args}},
    const double *params
){
    // uint offset =  blockDim.y * blockDim.x * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;
    uint offset = blockDim.x * blockIdx.x + threadIdx.x;

    if(offset >= {{thread_length}}) {
        // printf("up boundary: %d\n", {{thread_length}});
        return;
        }

    dcmplx k1[{{mat_size}}];
    dcmplx k2[{{mat_size}}];
    dcmplx k3[{{mat_size}}];
    dcmplx temp[{{mat_size}}];
    dcmplx rho[{{mat_size}}];

    for(uint row = 0; row < {{dim}}; row++){
        for(uint col = 0; col < {{dim}}; col++){
            rho[row*{{dim}} + col] = rho0[row*{{dim}} + col];
        }
    }

    uint step_offset = offset%{{ranged_param_len}};
    const double step = step_list[step_offset];
    //  double step = 0.001;

    {% for code in dim_convert %}
        {{code}}
    {% endfor %}

    double t = 0.0;
    uint index = 0;
    for(int i=0; i<{{moments}}; i++){
        clear_matrix(k1);
        clear_matrix(k2);
        clear_matrix(k3);
        clear_matrix(temp);
        
        //k1
        t = step*i;
        {%for code in effect_of_Hamiltonian_k1 %}
            {{ code }}
        {% endfor %}
        //

        {% for code in effect_of_Lindblad_k1 %}
            {{ code }}
        {% endfor %}

        add_AB(rho, k1, temp, 1.0, 0.5*step); //temp = rho +0.5h*k1

        //k2
        t += 0.5*step;
        {%for code in effect_of_Hamiltonian_k2 %}
            {{ code }}
        {% endfor %}
        
        {% for code in effect_of_Lindblad_k2 %}
            {{ code }}
        {% endfor %}

        //

        add_ABC(rho, k1, k2, temp, 1.0, -step, 2.0*step); //temp = rho-h*k1+2h*k2
        //k3
        t += 0.5*step;
        {%for code in effect_of_Hamiltonian_k3 %}
            {{ code }}
        {% endfor %}
        //

        {% for code in effect_of_Lindblad_k3 %}
            {{ code }}
        {% endfor %}

        sum_ABC(k1, k2, k3, rho, step/6.0, step*4.0/6.0, step/6.0); //rho += h*(k1+4*k2+k3)/6

        uint u = {{moments}}/{{tlength}};
        if (i%u == 0){
            {% for code in expects %}
                {{ code }}
            {% endfor %}
        index++;
        }
    }

}