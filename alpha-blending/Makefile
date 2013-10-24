INSTALL := $(HOME)/qrc/track/opencv/install
LIBS :=  -lopencv_core -lopencv_highgui -lopencv_imgproc

CXXFLAGS := -g -O0
CXXFLAGS += -I$(INSTALL)/include
CXXFLAGS += -L$(INSTALL)/lib $(LIBS)

LOGOS := ../resources/LinuxLogo.jpg ../resources/WindowsLogo.jpg

main: alphaBlend

color: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./alphaBlend $(LOGOS)

test: color

clean:
	rm -rf alphaBlend *.dSYM

debug: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	lldb ./alphaBlend -- $(LOGOS)

.PHONY: main color test clean debug
