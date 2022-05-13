all : figs/graph.png figs/graph.pdf data/1976-2000_ASCII.txt.gz

data/1976-2000_ASCII.txt.gz :
	wget -O $@ http://koeppen-geiger.vu-wien.ac.at/data/1976-2000_ASCII.txt.gz

%.png : %.dot
	dot -o $@ -Tpng $<

%.pdf : %.dot
	dot -o $@ -Tpdf $<
