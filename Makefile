all : figs/graph.png figs/graph.pdf data/1976-2000_ASCII.txt.gz figs/map.pdf
	mkdir -p diagnostic_figures

data/1976-2000_ASCII.txt.gz :
	wget -O $@ http://koeppen-geiger.vu-wien.ac.at/data/1976-2000_ASCII.txt.gz

%.png : %.dot
	dot -o $@ -Tpng $<

figs/graph.pdf : graph.dot
	dot -o $@ -Tpdf $<

figs/map.pdf : map.py
	python3 $<
