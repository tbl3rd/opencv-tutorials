INSTALL := ../opencv/install
LIBS := \
-lopencv_ml \
-lopencv_highgui \
-lopencv_core \
#

CXXFLAGS := -g -O0
CXXFLAGS += -I$(INSTALL)/include
CXXFLAGS += -L$(INSTALL)/lib $(LIBS)

EXECUTABLE := old-svmNonlinear
EXECUTABLE := svmNonlinear

main: $(EXECUTABLE)

help: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH ./$(EXECUTABLE)

test: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./$(EXECUTABLE) $(ARGS)

clean:
	rm -rf $(EXECUTABLE) *.dSYM

debug: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	lldb ./$(EXECUTABLE) -- $(ARGS)

.PHONY: main help test clean debug


# cv::imwrite(cv::String const&, cv::_InputArray const&, std::__1::vector<int, std::__1::allocator<int> > const&)
# cv::imwrite(cv::String const&, cv::_InputArray const&, std::vector<int, std::allocator<int> > const&)
# cv::imwrite(cv::String const&, cv::_InputArray const&, std::vector<int, std::allocator<int> > const&)
# cv::imwrite(cv::String const&, cv::_InputArray const&, std::vector<int, std::allocator<int> > const&)
# cv::imwrite(cv::String const&, cv::_InputArray const&, std::vector<int, std::allocator<int> > const&)
# cv::imwrite(cv::String const&, cv::_InputArray const&, std::vector<int, std::allocator<int> > const&)
# cv::imwrite(cv::String const&, cv::_InputArray const&, std::vector<int, std::allocator<int> > const&)
