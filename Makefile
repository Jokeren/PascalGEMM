BINS := sgemm_tn_128x128_vec sgemm_tn_128x128 sgemm_nn_128x128_vec sgemm_nn_128x128 sgemm_nt_128x128_vec sgemm_nt_128x128
TARGETS := $(addsuffix .cubin, $(BINS))
TEMPLATES := $(addsuffix _template.cubin, $(BINS))

all: $(BINS) sgemm

$(BINS):
	nvcc -arch sm_61 -m 64 $@.cu -cubin -O3 -o $@_template.cubin
	maxas.pl -i $@.sass $@_template.cubin $@.cubin

sgemm: sgemm.cu
	nvcc $^ -o $@ -O3 -arch sm_61 -lcuda -lcudart

clean:
	rm $(TARGETS) $(TEMPLATES) sgemm

.PHONY:
	all clean

#utils
print-% : ; $(info $* is $(flavor $*) variable set to [$($*)]) @true   
