/* File:   genericTsPool.h
 * Authors: Kostis Papadakis (kpapadakis@protonmail.com) (2022)
 * Description: A thread safe generic memory pool that is memory type agnostic
 *              which means that it can manage CUDA,HIP,Stack,Dynamic memory et
 al.
 * This file defines the following class:
 *       GENERIC_TS_POOL::MemPool
 *
        //main.cpp
        Example Minimal Usage:
        size_t sz=1<<12;
        char block[sz];   <-- this can be whatever memory block
        GENERIC_TS_POOL::MemPool pool(&block,sz);

        //Allocate an array of 4 doubles
        double* number = pool.allocate<double>(4);

        // Do smth with it
        number[0]=1.0;
        number[1]=2.0;
        .  .  .
        .  .  .
        //Deallocate the array
        pool.deallocate(number);

        //Defrag the pool.
        pool.defrag();


 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
 USA.
 * */
#pragma once
#include <atomic>
#include <cstddef>
#include <iostream>
#include <map>
#include <mutex>
#include <stdlib.h>
#include <unordered_map>
#include <utility>

#define THREADSAFE

namespace GENERIC_TS_POOL {
class MemPool {
   struct AllocHeader {
      size_t size;
      size_t padding;
   };

private:
   // private members
   void* _memory;
   size_t _bytes;
   size_t _freeSpace;
   std::atomic<int> users = {0};
   mutable std::mutex _mlock;
   std::map<size_t, size_t> _freeBlocks;
   std::unordered_map<size_t, AllocHeader> _allocBlocks;
   float HWM = 0;
   //~ private members

   inline void _lock() const noexcept {
#ifdef THREADSAFE
      _mlock.lock();
#endif
   }

   inline void _unlock() const noexcept {
#ifdef THREADSAFE
      _mlock.unlock();
#endif
   }

   [[nodiscard]] inline size_t calculatePadding(size_t size, size_t alignment) const noexcept {
      size_t remainder = size % alignment;
      if (remainder == 0) {
         return 0;
      }
      return alignment - remainder;
   }

   void reset() noexcept {
      _lock();
      _freeBlocks.clear();
      _allocBlocks.clear();
      _freeBlocks[reinterpret_cast<size_t>(_memory)] = capacity();
      _freeSpace = capacity();
      _unlock();
   }

   void reset_unsafe() noexcept {
      _freeBlocks[reinterpret_cast<size_t>(_memory)] = capacity();
      _freeSpace = capacity();
   }

public:
   MemPool(void* block, size_t maxSize) : _bytes(maxSize), _freeSpace(maxSize) {
      if (block == nullptr) {
         throw std::runtime_error("Null pointer cannot be used to instantiate mempool!");
      }
      _memory = block;
      reset();
   }
   MemPool() : _memory(nullptr) {}
   MemPool(const MemPool& other) = delete;
   MemPool(MemPool&& other) = delete;
   MemPool& operator=(const MemPool& other);
   MemPool& operator=(MemPool&& other) = delete;
   ~MemPool() = default;

   inline float memory_hwm(void) const noexcept { return HWM; }
   inline size_t capacity(void) const noexcept { return _bytes; }
   inline size_t size(void) const noexcept { return _bytes - _freeSpace; }
   inline float load(void) const noexcept { return static_cast<float>(size()) / static_cast<float>(capacity()); }
   inline void release(void) noexcept { reset(); }

   void resize(void* block, size_t maxSize) {
      if (block == nullptr) {
         throw std::runtime_error("Null pointer cannot be used to resize mempool!");
      }
      _memory = block;
      _bytes = maxSize;
      _freeSpace = maxSize;
      reset();
   }

   template <typename F>
   void init(size_t maxSize, F&& allocFunction) {
      users += 1;
      _lock();
      if (_memory != nullptr) {
         _unlock();
         return;
      }
      _memory = allocFunction(maxSize);
      _bytes = maxSize;
      _freeSpace = maxSize;
      reset_unsafe();
      _unlock();
   }

   void stats() noexcept {
      printf("*********************************\n");
      printf("Capacity=%zu , size= %zu , load = %f\n ", capacity(), size(), load());
      printf("FreeSlots\n");
      for (std::map<size_t, size_t>::iterator slot = _freeBlocks.begin(); slot != _freeBlocks.end(); slot++) {
         // printf("\tBlock %zu with size %zu\n", slot->first, slot->second);
         std::cout << "\tBlock " << slot->first << " with size " << slot->second << std::endl;
      }
      printf("\n");
      printf("\n");
      printf("AllocSlots\n");
      for (std::unordered_map<size_t, AllocHeader>::iterator slot = _allocBlocks.begin(); slot != _allocBlocks.end();
           slot++) {
         printf("\tBlock %zu with size = %zu and padding= %zu\n", slot->first, slot->second.size, slot->second.padding);
      }
      printf("*********************************\n");
   }

   void defrag() {
      _lock();
      auto left = _freeBlocks.begin();
      while (left != _freeBlocks.end()) {
         auto right = left;
         right++;
         if (right == _freeBlocks.end()) {
            _unlock();
            return;
         }

         if (left->first + left->second == right->first) {
            std::pair<size_t, size_t> newBlock{left->first, left->second + right->second};
            _freeBlocks.erase(left);
            _freeBlocks.erase(right);
            auto ok = _freeBlocks.insert(newBlock);
            if (ok.second) {
               left = ok.first;
            } else {
               throw std::runtime_error("Something caused a catastrophic failure during defragmentation");
            }
         } else {
            left++;
         }
      }
      _unlock();
      return;
   }

   std::map<size_t, size_t>::iterator findBlock(size_t& bytes, size_t alignment) {
      for (std::map<size_t, size_t>::iterator slot = _freeBlocks.begin(); slot != _freeBlocks.end(); slot++) {
         size_t baseAddress = slot->first;
         size_t padding = calculatePadding(baseAddress, alignment);
         if (slot->second >= bytes + padding) {
            bytes += padding;
            return slot;
         }
      }
      return _freeBlocks.end();
   }

   template <typename T>
   [[nodiscard]] T* allocate(const size_t elements) noexcept {

      if (elements == 0) {
         return nullptr;
      }
      const size_t bytesToAllocate = elements * sizeof(T);
      const size_t alignment = std::max(8ul, std::alignment_of<T>::value);
      size_t allocationSize = bytesToAllocate;

      if (bytesToAllocate > _freeSpace) {
         std::cerr << "Not enough space to allocate " << bytesToAllocate << std::endl;
         std::cerr << "Free space is " << _freeSpace << std::endl;
         return nullptr;
      }

      _lock();
      auto candidate = findBlock(allocationSize, alignment);
      size_t pad = allocationSize - bytesToAllocate;
      if (candidate == _freeBlocks.end()) {
         _unlock();
         return nullptr;
      }

      // Handle Padding
      size_t baseAddress = candidate->first + pad;

      // Split blocks
      _allocBlocks.insert(std::pair<size_t, AllocHeader>{baseAddress, AllocHeader{allocationSize, pad}});
      _freeSpace -= allocationSize;
      if (_freeSpace > 0) {
         size_t newBlock = candidate->first + allocationSize;
         size_t newBlockSize = candidate->second - allocationSize;
         _freeBlocks.insert(std::pair<size_t, size_t>{newBlock, newBlockSize});
      }
      _freeBlocks.erase(candidate);
      HWM = std::max(HWM, load());
      _unlock();
      return reinterpret_cast<T*>(baseAddress);
   }

   bool deallocate(void* ptr) noexcept {
      if (ptr == nullptr) {
         return true;
      }

      _lock();
      auto it = _allocBlocks.find(reinterpret_cast<size_t>(ptr));
      if (it == _allocBlocks.end()) {
         _unlock();
         std::cerr << "Could not find alloced block to deallocate!" << std::endl;
         return false;
      }

      size_t pad = it->second.padding;
      size_t baseAddress = it->first - pad;
      _freeBlocks[baseAddress] = it->second.size;
      _freeSpace += it->second.size;
      if (it->second.size == 0) {
         abort();
      }
      _allocBlocks.erase(it);
      _unlock();
      return true;
   }
   template <typename F>
   void destroy_with(F&& f) {
      if (users == 1) {
         f(_memory);
         _memory = nullptr;
      }
      users--;
   }
};
} // namespace GENERIC_TS_POOL
