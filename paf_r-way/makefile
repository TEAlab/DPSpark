all: libscalers.a
	CFLAGS="-I." LDFLAGS="-L." python setup.py build_ext -i

libscalers.a:
	gcc -fPIC -fopenmp -c paf_rdp_omp.c -o paf_rdp_omp.o
	ar cr libpafomp.a paf_rdp_omp.o

	# ar cr $@ $^

# %.o:
# 	gcc -fPIC –fopenmp -c $< -o $@

clean:
	rm  *.a *.so

# gcc -c scalars.c -o scalars.o
# gcc -fPIC -c scalers.c -o scalers.o
# ar cr libscalers.a scalers.o
