
'''
MPI Assignements
'''
from mpi4py import MPI

# Exercise 1 : hello world

# Question 1 :
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

print('hello world',RANK,SIZE)

# Question 2 :

print('hello world from the processor {:d} out of {:d}'.format(RANK, SIZE))

# Question 3:

if RANK == 0:
    print('Hello world from processor {:d} out of {}'.format(RANK,SIZE))

# Exercise 2 : sharing data
while True :
    if RANK == 0 :
        data = int(input('enter a number : '))
    else :
        data = None

    data = COMM.bcast(data , root = 0)
    
    if data < 0 :
        break

    print(f"The Process {RANK} got the data {data}")

    COMM.Barrier()


# Exercise 3 : Sending in a ring

if RANK == 0:
    data = int(input('Entrer unevaleur: '))
else :
    data = None

for i in range(SIZE):
    if RANK == i:
        data = data + RANK
        print('Le processeur {} a {}'.format(RANK,data))
        next_rank = RANK + 1
        COMM.send(data , dest = next_rank)
    else :
        prev_rank = RANK -1

    data = COMM.recv(source = prev_rank)     

# Exercise 4 : Scattering matrix
import numpy as np

m = 8
n = 8 

if RANK ==0:
    A = np.random.randint(10,size=(n,m))
else :
    A = np.zeros((m,n))
    
A_rank1 = []
for i in range(3):
    for j in range(4,7):
        A_rank1.append(A[i,j])

A_rank2 = []
for i in range(4 , 7):
    for j in range(7):
        A_rank2.append(A[i,j])

        
A_rank3 = []
for i in range(4,7):
    for j in range(4,7):
        A_rank3.append(A[i,j])

sendbuff = [A,A_rank1,A_rank2,A_rank3]
recvbuf = COMM.scatter(sendbuff, root = 0)

print('process ',RANK, 'recieved ',recvbuf)

# Exercise 5 :
# in MatrixVectorMult.py

# Exercise 6 :

# question 1 :

N = 840

def compute_pi(N):
    sum = 0
    for i in range(1,N):
        sum += 1/(1+((i-0.5)/N)**2)
    return (4/N)*sum

print('Value of pi : {} from processor {} '.format(compute_pi(N),RANK))

# question 2:

if RANK ==0:
    sum = 0
    for i in range(1,420-1):
        sum += 1/(1+((i-0.5)/N)**2)
    print('the partial sum by process 0 is : ',(4/N)*sum)
    COMM.send((4/N) *sum,dest = 1)
if RANK == 1:
    recv = COMM.recv(source = 0)
    for i in range(420,840):
        recv += (4/N)* 1/(1+((i-0.5)/N)**2)
    print('the sum of the two results is ')
    print('The value of pi : ',recv) 

# part 2
import numpy as np
N = 8400 # Increasing the value of N generally increases the execution time
start_time = MPI.Wtime()

local_n = int(N / SIZE)
local_sum = 0.0
for i in range(RANK * local_n + 1, (RANK + 1) * local_n + 1):
    x = (i - 0.5) / N
    local_sum += 4.0 / (1.0 + x**2)

partial_sums = COMM.gather(local_sum, root=0)

if RANK == 0:
    pi = np.sum(partial_sums) / N
    end_time = MPI.Wtime()
    print("Approximate value of pi:", pi)
    print("Execution time:", end_time - start_time)
else:
    COMM.send(local_sum, dest=0)

if RANK == 0:
    for i, partial_sum in enumerate(partial_sums):
        print("Partial sum for rank", i, ":",partial_sum)
