/* File: qdargparser.h
A sub 100LOC quick and dirty parser.
It's not great code but works!
Author: Kostis Papadakis, 2025 (kpapadakis@protonmail.com)
Usage:
  constexpr auto delim=' '; //for no delim use '0'
  QdArgParser<delim> cli_args(argc, argv);
  for (auto it=cli_args.begin(); it!=cli_args.end();++it){
    std::cout<<*it<<"\n";
  }
This program is free software; you can redistribute it and/or
modify it as you wish. You can make money with it or scam people.
*/
#include <cstring>
#include <iostream>
#include <iterator>
#include <stdlib.h>
#include <string_view>
#include <vector>
template <char _delim, typename Allocator = std::allocator<char>>
struct QdArgParser {
  std::vector<char, Allocator> _data;
  QdArgParser(int argc, char **buffer) {
    if (argc > 0) {
      for (int i = 0; i < argc; ++i) {
        _data.insert(_data.end(), buffer[i],
                     buffer[i] + std::strlen(buffer[i]) + 1);
      }
      std::replace(_data.begin(), _data.end(), '\0', ' ');
      _data.pop_back();
    }
  }
  QdArgParser(const char *buffer, std::size_t size) {
    if (size > 0) {
      _data.insert(_data.end(), buffer, buffer + size + 1);
      std::replace(_data.begin(), _data.end(), '\0', ' ');
      _data.pop_back();
    }
  }
  template <typename F> auto apply(F &&functor) const noexcept {
    return functor(_data.begin(), _data.end());
  }
  template <typename T, typename F>
  std::vector<T> apply_collect_per_element(F &&functor) noexcept {
    std::vector<T> vals;
    for (auto it = begin(); it != end(); it = it.next()) {
      vals.push_back(functor(*it));
    }
    return vals;
  }
  void print() noexcept {
    std::cout << "{";
    for (auto it = begin(); it != end(); ++it) {
      std::cout << *it << ", ";
    }
    std::cout << "}\n";
  }
  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    std::string_view view;
    char *_pos;
    std::size_t shift={0};
    std::size_t left_over={0};
    Iterator(char *pos, std::size_t len) : _pos(pos) {
      const char *it = (_delim == '0') ? pos + 1 : std::find(pos, pos + len, _delim);
      if (it == pos + len) {
        view = std::string_view(pos, pos + len);
        left_over = 0;
        shift = len;
      } else {
        view = std::string_view(pos, it);
        shift = it - pos;
        left_over = len - shift - 1;
      }
    }
    std::string_view operator*() { return view; }
    char *operator->() { return view.data(); }
    bool operator==(const Iterator &other) const noexcept {
      return _pos == other._pos;
    }
    bool operator!=(const Iterator &other) const noexcept {
      return !(_pos == other._pos);
    }
    char *data() { return _pos; }
    Iterator next() {
      return (_delim == '0')
                 ? Iterator(_pos + shift, left_over)
                 : Iterator(_pos + shift + 1 * (left_over > 0), left_over);
    }
    Iterator &operator++() {
      return *this=next(),*this;
    }
  };
  Iterator begin() { return Iterator(_data.data(), _data.size()); }
  Iterator end() { return Iterator(_data.data() + _data.size(), 0); }
};
