
OBJS = main.o LinguisticSet.o LinguisticDomain.o MembershipFunction.o FuzzyValue.o

main:  	$(OBJS)
	g++ -o main $(OBJS)

clean:
	rm *.o

main.o: 		main.cc
LinguisticSet.o:	LinguisticSet.cc LinguisticSet.hh
LinguisticDomain.o:	LinguisticDomain.cc LinguisticDomain.hh
MembershipFunction.o:	MembershipFunction.cc MembershipFunction.hh
FuzzyValue.o:		FuzzyValue.cc FuzzyValue.hh




