INSTALL := $(HOME)/qrc/track/opencv/install
LIBS :=  -lopencv_core -lopencv_highgui -lopencv_imgproc

CXXFLAGS := -g -O0
CXXFLAGS += -I$(INSTALL)/include
CXXFLAGS += -L$(INSTALL)/lib $(LIBS)

EXECUTABLE := howToScanImages
IMAGEFILE := ../resources/Twas_Ever_Thus500.jpg

main: $(EXECUTABLE)

gray: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./$(EXECUTABLE) $(IMAGEFILE) 200 g

color: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./$(EXECUTABLE) $(IMAGEFILE) 200

test: gray color

clean:
	rm -rf $(EXECUTABLE) *.dSYM

debug: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	lldb ./$(EXECUTABLE) -- $(IMAGEFILE) 200

.PHONY: main help gray color test clean debug
