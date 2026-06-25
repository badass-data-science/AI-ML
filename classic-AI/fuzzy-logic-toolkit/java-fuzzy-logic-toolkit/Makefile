SRC = $(wildcard src/fuzzy/*.java)

run: compile
	java -cp out fuzzy.Main

compile:
	mkdir -p out
	javac -d out $(SRC)

clean:
	rm -rf out
