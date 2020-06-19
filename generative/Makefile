TPL=../github.tpl
SRC=$(wildcard *.ipynb)
DST=$(SRC:.ipynb=.html)


all: $(DST)

%.html: %.ipynb
	jupyter-nbconvert --to html --template=$(TPL) $<
