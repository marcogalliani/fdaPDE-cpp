#include <gtest/gtest.h>

#include "ScalarFieldTest.cpp"
#include "VectorFieldTest.cpp"

int main(int argc, char **argv){
  // start testing
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}