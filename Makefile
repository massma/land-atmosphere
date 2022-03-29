all : graph.png

%.png : %.dot
	dot -o $@ -Tpng $<
