#include<mpi.h>
#include<stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define CONVERGENCE_COEF 100

/**
 * Run: mpirun -np 12 ./a.out
 */

static int num_workers;
static int **topology;

static int *nr_workers;

static int *V;
static int N;

static int parent_id;


void init(int numProcs) {
	topology = malloc(3 * sizeof(int*));
	for (int i = 0; i < 3; i++) {
		topology[i] = calloc(numProcs, sizeof(int));
	}

	nr_workers = calloc(3, sizeof(int));

	parent_id = -1;
}

void initVec(int dim) {
	N = dim;
	V = calloc(N, sizeof(int));

	for (int i = 0; i < N; i++) {
		V[i] = i;
	}
}

void printTopology(int rank, int numProcs) {
	for (int j = 0; j < numProcs; j++) {
		printf("%d ", topology[rank][j]);
	}
	printf("\n");
}

void read_topology(int rank, int numProcs) {
    FILE *fp;
    char file_name[100];
    sprintf(file_name, "cluster%d.txt", rank);

    fp = fopen(file_name, "r");
	fscanf(fp, "%d", &num_workers);
	nr_workers[rank] = num_workers;
	for (size_t i = 1; i <= num_workers; i++) {
		int worker;
		fscanf(fp, "%d", &worker);
		topology[rank][worker] = i;
	}
}

void print_topology(int rank, int nProcesses) {
	printf("%d -> ", rank);
	for (int i = 0; i < 3; i++) {
		int k = 0;
		for (int j = 0; j < nProcesses; j++) {
			if (j == 0) {
				printf("%d:", i);
			}
			if (topology[i][j] != 0) {
				if (nr_workers[i] == 1) {
					printf("%d ", j);
				} else {
					if (++k < nr_workers[i]) {
						printf("%d,", j);
					} else {
						printf("%d ", j);
						k = 0;
					}
				}
			}
		}
	}
	printf("\n");
}

void print_result() {
	printf("Rezultat: ");
	for (int i = 0; i < N; i++) {
		printf("%d ", V[i]);
	}
	printf("\n");
}

void topology_establishment(int rank, int nProcesses, MPI_Status status) {
	if (rank < 3) {
		read_topology(rank, nProcesses);
		
		//trimit informatia topologiei celorlalte doua noduri cooronator
		for (int i = 0; i < 3; i++) {
			if (i != rank) {
				MPI_Send(topology[rank], nProcesses, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&nr_workers[rank], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				printf("M(%d,%d)\n", rank, i);
			}
		}

		// primesc informatia de la celelalte doua procese coordonator
		for (int i = 0; i < 3; i++) {
			if (i != rank) {
				MPI_Recv(topology[i], nProcesses, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(&nr_workers[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
			}
		}

		// trimit topologia workerilor
		for (int i = 0; i < nProcesses; i++) {
			if (topology[rank][i] != 0) {
				for (int j = 0; j < 3; j++) {
					MPI_Send(topology[j], nProcesses, MPI_INT, i, 0, MPI_COMM_WORLD);
					printf("M(%d,%d)\n", rank, i);
				}
				MPI_Send(nr_workers, 3, MPI_INT, i, 0, MPI_COMM_WORLD);
			}
		}
	} else {

		// workerii primesc topologia
		for (int i = 0; i < 3; i++) {
			MPI_Recv(topology[i], nProcesses, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		}

		MPI_Recv(nr_workers, 3, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		parent_id = status.MPI_SOURCE;	
	}
}

void perform_calculations(int rank, int arg, int nProcesses, MPI_Status status) {
	if (rank == 0) {
		initVec(arg);

		// procesul 0 calculeaza offsetul pentru fiecare worker
		int num_all_workers = 0;
		for (int i = 0; i < 3; i++) {
			num_all_workers += nr_workers[i];
		}

		int offset = N / num_all_workers;
		int diff = N % num_all_workers;

		// daca nu se va imparti exact la numarul total de workeri
		// gasesc clusterul cu cei mai multi workeri
		int max = 0;
		int maxCluster;
		for (int i = 0; i < 3; i++) {
			if (nr_workers[i] >= max) {
				max = nr_workers[i];
				maxCluster = i;
			}
		}

		// cu VPosition voi stii mereu la ce pozitie ma aflu in vector
		int VPosition = 0;
		// trimit workerilor lui 0
		for (int i = 0; i < nProcesses; i++) {
			int sentDiff = 0;	// flag care ne asigura ca trimit diferenta doar primului worker din clusterul cu nr maxim de workeri
			if (topology[rank][i] != 0) {
				int size = 0;
				if (maxCluster == rank) {
					if (!sentDiff) {
						size = offset + diff;
						sentDiff = 1;
					} else {
						size = offset;
					}
				} else {
					size = offset;	
				}

				MPI_Send(&size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(V + VPosition, size, MPI_INT, i, 0, MPI_COMM_WORLD);
				VPosition += size;
			}
		}

		// trimit clusterelor 1 si 2
		int sentDiff = 0;
		for (int j = 1; j < 3; j++) {
			int size;
			if (maxCluster == j) {
				size = offset * nr_workers[j] + diff;
			} else {
				size = offset * nr_workers[j];
			}

			MPI_Send(&size, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
			MPI_Send(V + VPosition, size, MPI_INT, j, 0, MPI_COMM_WORLD);
			VPosition += size;
		}

		// parintele 0 face recv de la workeri
		VPosition = 0;
		for (int i = 0; i < nProcesses; i++) {
			if (topology[0][i] != 0) {
				int offset;
				MPI_Recv(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(V + VPosition, offset, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
				VPosition += offset;
			}
		}

		// 0 primeste rezultatele de la celelalte 2 clustere
		for (int i = 1; i < 3; i++) {
			int offset;
			MPI_Recv(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
			MPI_Recv(V + VPosition, offset, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
			VPosition += offset;
		}

		// printez rezultatul final
		print_result();
		
	}

	// daca este un worker al lui 0
	if (rank > 2 && topology[0][rank] != 0) {
		// primesc o bucata de vector
		int recvSize;
		int *recvVec;
		MPI_Recv(&recvSize, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		recvVec = calloc(recvSize, sizeof(int));
		MPI_Recv(recvVec, recvSize, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

		// modific bucata
		for (int i = 0; i < recvSize; i++) {
			recvVec[i] *= 2;
		}

		// o trimit inapoi la 0
		MPI_Send(&recvSize, 1, MPI_INT, parent_id, 0, MPI_COMM_WORLD);
		MPI_Send(recvVec, recvSize, MPI_INT, parent_id, 0, MPI_COMM_WORLD);
	}
	
	// procesele coordonator 1 si 2 vor primi bucatiile de vector asignate lor
	if (rank == 1 || rank == 2) {

		// fac recv la bucata de vector a clusterului
		int recvSize;
		int *recvVec;
		MPI_Recv(&recvSize, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		recvVec = calloc(recvSize, sizeof(int));
		MPI_Recv(recvVec, recvSize, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

		// impart si trimit bucatile la workeri
		int offset =  recvSize / num_workers;
		int diff = recvSize % num_workers;

		int VPosition = 0;
		int sentDiff = 0;
		for (int i = 0; i < nProcesses; i++) {
			if (topology[rank][i] != 0) {
				int size = 0;
				if (!sentDiff) {
					size = offset + diff;
					sentDiff = 1;
				} else {
					size = offset;
				}
				
				MPI_Send(&size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(recvVec + VPosition, size, MPI_INT, i, 0, MPI_COMM_WORLD);

				VPosition += size;
			}
		}

		// clusterul primeste vectorul modificat de workeri
		VPosition = 0;
		for (int i = 0; i < nProcesses; i++) {
			if (topology[rank][i] != 0) {
				int offset;
				MPI_Recv(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(recvVec + VPosition, offset, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
				VPosition += offset;
			}
		}

		// clusterele vor trimite rezultatele catre 0
		MPI_Send(&VPosition, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Send(recvVec, VPosition, MPI_INT, 0, 0, MPI_COMM_WORLD);
		
	}

	// workerii clusterelor 1 si 2 fac recv si trimit mai departe cu modificarile facute
	if (rank >= 3 && (topology[1][rank] != 0 || topology[2][rank] != 0)) {
		int recvSize;
		int *recvVec;
		MPI_Recv(&recvSize, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
		recvVec = calloc(recvSize, sizeof(int));
		MPI_Recv(recvVec, recvSize, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

		for (int i = 0; i < recvSize; i++) {
			recvVec[i] *= 2;
		}

		MPI_Send(&recvSize, 1, MPI_INT, parent_id, 0, MPI_COMM_WORLD);
		MPI_Send(recvVec, recvSize, MPI_INT, parent_id, 0, MPI_COMM_WORLD);

	}
}

int main(int argc, char * argv[]) {
	int rank, nProcesses;
	int *parents;

	MPI_Init(&argc, &argv);
	MPI_Status status;
	MPI_Request request;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

	init(nProcesses);

	char finalString[100];

	// -------Stabilirea topologiei------------
	topology_establishment(rank, nProcesses, status);	
	
	// printez topologia acum cunoscuta
	print_topology(rank, nProcesses);

	
	// -------Realizarea calculelor------------
	perform_calculations(rank, atoi(argv[1]), nProcesses, status);


	MPI_Finalize();
	return 0;
}