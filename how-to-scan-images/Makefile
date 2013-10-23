INSTALL := $(HOME)/qrc/track/opencv/install
LIBS :=  -lopencv_core -lopencv_highgui -lopencv_imgproc

CXXFLAGS := -g -O0
CXXFLAGS += -I$(INSTALL)/include
CXXFLAGS += -L$(INSTALL)/lib $(LIBS)

main: howToScanImages

gray: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./howToScanImages ../resources/Twas_Ever_Thus500.jpg 200 g

color: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./howToScanImages ../resources/Twas_Ever_Thus500.jpg 200

test: gray color

clean:
	rm -rf howToScanImages *.dSYM

debug: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	lldb ./howToScanImages

.PHONY: main gray color test clean debug
