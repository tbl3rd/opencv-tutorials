INSTALL := $(HOME)/qrc/track/opencv/install
LIBS :=  -lopencv_core -lopencv_highgui -lopencv_imgproc

CXXFLAGS := -g -O0
CXXFLAGS += -I$(INSTALL)/include
CXXFLAGS += -L$(INSTALL)/lib $(LIBS)


test: loadAndDisplayImage
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./loadAndDisplayImage ../resources/Twas_Ever_Thus500.jpg

clean:
	rm -rf loadAndDisplayImage gray-image.jpg *.dSYM

debug: loadAndDisplayImage
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	lldb ./loadAndDisplayImage ../resources/Twas_Ever_Thus500.jpg
