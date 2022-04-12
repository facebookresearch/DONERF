


#ifndef INCLUDED_UTILS_IO_H
#define INCLUDED_UTILS_IO_H

#pragma once


template <typename T>
T read(std::istream& in)
{
  T value;
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  return value;
}

template <typename T>
std::istream& read(std::istream& in, T& value)
{
  return in.read(reinterpret_cast<char*>(&value), sizeof(T));
}

template <typename T>
std::istream& read(std::istream& in, T* values, size_t count = 1)
{
  return in.read(reinterpret_cast<char*>(values), sizeof(T) * count);
}

template <typename T, size_t N>
std::istream& read(std::istream& in, T (&values)[N])
{
  return in.read(reinterpret_cast<char*>(values), sizeof(values));
}

template <typename T>
std::ostream& write(std::ostream& out, T value)
{
  return out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
std::ostream& write(std::ostream& out, const T* values, size_t count)
{
  return out.write(reinterpret_cast<const char*>(values), sizeof(T) * count);
}

template <typename T, size_t N>
std::ostream& write(std::ostream& out, const T (&values)[N])
{
  return out.write(reinterpret_cast<const char*>(values), sizeof(values));
}

template <typename T, size_t N>
std::ostream& writeString(std::ostream& out, const T (&values)[N])
{
  return out.write(reinterpret_cast<const char*>(values), sizeof(values) - 1);
}

#endif  // INCLUDED_UTILS_IO_H
