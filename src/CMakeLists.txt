# better way ?
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

# Source files
# ====================================
#           (Glob is not recommended)

SET(SCAN3D_HDS
    leopard.hpp
	triangulation.hpp
	paths.hpp
)

SET(SCAN3D_SRC
    leopard.cpp
	triangulation.cpp
)

# main library (prevent recompilation when building tests)
ADD_LIBRARY(scan3d ${SCAN3D_SRC} ${SCAN3D_HDS})
TARGET_LINK_LIBRARIES(scan3d ${OpenCV_LIBS})
SET_PROPERTY(TARGET scan3d PROPERTY CXX_STANDARD 14)


#  main executable
# ==========================

ADD_EXECUTABLE(main main.cpp)
TARGET_LINK_LIBRARIES(main scan3d ${OpenCV_LIBS})
SET_PROPERTY(TARGET main PROPERTY CXX_STANDARD 14)



