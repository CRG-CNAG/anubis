# FASTQINS
FASTQINS is a Python pipeline to map transponson insertions from Tn-seq data. 

## Requirements
Specific libraries are required by ANUBIS. We provide a [requirements](./requirements.txt) file to install everything at once. To do so, you will need first to have [pip](https://pip.pypa.io/en/stable/installing/) installed and then run:

```bash
sudo apt-get install python-pip    # if you need to install pip
pip install -r requirements.txt
```

## Installation & Help

Download this repository and run:

```bash
python3 setup.py install
```

You may require to call it using sudo. Once installed, `anubis` will be integrated in your python distribution.

## Example

Requirements to run an experiment are: 

  -i [fastq files with transposon mapped, if no -i2 is passed, single-end mapping by default] <br />
  -t [IR transposon sequence, expected to be found contiguous genome sequence] <br />
  -g [genome sequence, fasta or genbank format]  <br />
  -o [output directory to locate the results]

As example, we included a pair of files that you can use to test the pipeline functioning as:

```bash
fastqins -i ./test/test_read2.fastq.gz -i2 ./test/test_read2.fastq.gz -t TACGGACTTTATC -g ./test/NC_000912.fna -o test -v -r 0
```

## Contact

This project has been fully developed at [Centre for Genomic Regulation](http://www.crg.eu/) at the group of [Design of Biological Systems](http://www.crg.eu/en/luis_serrano).

If you experience any problem at any step involving the program, you can use the 'Issues' page of this repository or contact:

[Miravet-Verde, Samuel](mailto:samuel.miravet@crg.eu)       
[Lluch-Senar, Maria](mailto:maria.lluch@crg.eu)       
[Serrano, Luis](mailto:luis.serrano@crg.eu)

## License

ANUBIS is under a common GNU GENERAL PUBLIC LICENSE. Plese, check [LICENSE](./LICENSE) for further information.

###### [2020] - Centre de Regulació Genòmica (CRG) - All Rights Reserved*

