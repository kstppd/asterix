#include "../include/genericTsPool.h"
#include <cassert>
#include <gtest/gtest.h>
#include <iostream>
#include <random>

template <typename T>
bool allocate_to_the_brim_and_test(size_t len) {

   T* data = new T[len];
   if (!data) {
      std::cerr << "Failed to allocate initial buffer" << std::endl;
      return false;
   }

   std::random_device dev;
   std::mt19937 rng(dev());
   std::uniform_int_distribution<size_t> dist(1 << 2, len / 2);
   GENERIC_TS_POOL::MemPool p(data, len * sizeof(T));

   bool ok = true;
   int step = 0;
   std::vector<T*> ptrs;
   for (;;) {
      size_t sz = dist(rng);
      T* array = p.allocate<T>(sz);
      if (array == nullptr) {
         // printf("Reached max allocation with load = %f.\n",p.load());
         assert(p.load() < 1.0 && "erroneous load factor in pool!");
         if (step == 0) {
            step++;
            p.release();
            ptrs.clear();
            continue;
         } else {
            break;
         }
      }
      ptrs.push_back(array);

      // Set
      for (size_t i = 0; i < sz; ++i) {
         array[i] = i;
      }

      // Check
      for (size_t i = 0; i < sz; ++i) {
         if (array[i] != (T)i) {
            std::cout << "ERROR: Expected " << i << " got " << array[i] << std::endl;
            ok = false;
            break;
         }
      }
   }

   for (auto ptr : ptrs) {
      p.deallocate(ptr);
   }
   p.defrag();
   // Now we should have 0 alloc blocks and only one large degraged full free block
   ok = ok && (p.size() == 0);
   delete[] data;
   data = nullptr;
   return ok;
}

int main() {
   size_t test_size = 1 << 24;
   for (size_t i = 0; i < 10; ++i) {
      printf("Running test rep %zu/10...\n", i);
      auto ok1 = allocate_to_the_brim_and_test<float>(test_size);
      auto ok2 = allocate_to_the_brim_and_test<double>(test_size);
      auto ok3 = allocate_to_the_brim_and_test<int>(test_size);
      EXPECT_TRUE(ok1);
      EXPECT_TRUE(ok2);
      EXPECT_TRUE(ok3);
   }
   return 0;
}
