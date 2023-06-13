DATA_DIR=./data
RESULT_DIR=./results
RESULTS_EMBEDDING=${RESULT_DIR}/Embeddings

#Files
PROTEIN_DOMAINS = protein_domains.txt
PROTEIN_INTERACTIONS = protein_links.txt



all : ${RESULTS_EMBEDDING}/embedings_all_tms_nonredun_780.pkl

# setup data files and results directory

${DATA_DIR}/all_tms_noredun_780.fasta :
	mkdir -p data
	bash src/download_data.sh

${RESULTS_EMBEDDING}/embedings_all_tms_nonredun_780.pkl : ${Data_DIR}/all_tms_noredun_780.fasta
	mkdir -p ${RESULT_DIR}
	mkdir -p ${RESULT_EMBEDDING}
	python make_embedings.py -i ${DATA_DIR}/all_tms_noredun_780.fasta -o ${RESULTS_EMBEDDING}/embedings_all_tms_nonredun_780.pkl

#${RESULTS_EMBEDDING}/test3.pkl : ${DATA_DIR}/all_tms_noredun_780.fasta
#	mkdir -p ${RESULT_DIR}
#	mkdir -p ${RESULTS_EMBEDDING}
#	python src/make_embedings.py -i test/test3.fasta -o ${RESULTS_EMBEDDING}/test3.pkl

.PHONY: clean
clean:
	rm -r data
	rm *.sif
