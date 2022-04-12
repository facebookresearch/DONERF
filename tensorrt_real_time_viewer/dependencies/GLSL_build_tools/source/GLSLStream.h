


#ifndef INCLUDED_GLSLSTREAM_H
#define INCLUDED_GLSLSTREAM_H

#pragma once

#include <cstring>
#include <string>
#include <algorithm>

#include "Log.h"


namespace GLSL
{
  struct Token
  {
    const char* begin;
    const char* end;

    Token(const char* begin = nullptr)
      : begin(begin), end(begin)
    {
    }

    Token(const char* begin, const char* end)
      : begin(begin), end(end)
    {
    }

    Token(const char* begin, size_t length)
      : begin(begin), end(begin + length)
    {
    }
  };

  inline bool operator ==(const Token& a, const Token& b)
  {
    if (a.end - a.begin != b.end - b.begin)
      return false;
    for (auto c1 = a.begin, c2 = b.begin; c1 != a.end; ++c1, ++c2)
      if (*c1 != *c2)
        return false;
    return true;
  }

  inline bool operator ==(const Token& t, const char* str)
  {
    return std::strncmp(t.begin, str, t.end - t.begin) == 0;
  }

  inline bool operator ==(const char* str, const Token& t)
  {
    return t == str;
  }

  inline bool operator !=(const Token& a, const Token& b)
  {
    return !(a == b);
  }

  inline bool operator !=(const Token& t, const char* str)
  {
    return std::strncmp(t.begin, str, t.end - t.begin) != 0;
  }

  inline bool operator !=(const char* str, const Token& t)
  {
    return t != str;
  }

  enum OPERATOR
  {
    OP_INVALID,
    OP_PLUS,
    OP_MINUS,
    OP_ASTERISK,
    OP_SLASH,
    OP_CIRCUMFLEX,
    OP_TILDE,
    OP_LPARENT,
    OP_RPARENT,
    OP_LBRACKET,
    OP_RBRACKET,
    OP_LBRACE,
    OP_RBRACE,
    OP_QUEST,
    OP_DOT,
    OP_COLON,
    OP_COMMA,
    OP_SEMICOLON,
    OP_EQ,
    OP_EEQ,
    OP_NEQ,
    OP_LT,
    OP_LEQ,
    OP_LL,
    OP_GT,
    OP_GEQ,
    OP_GG,
    OP_AND,
    OP_AAND,
    OP_OR,
    OP_OOR,
    OP_BANG,
    OP_PERCENT
  };

  Token token(OPERATOR op);


  class Stream;

  class LexerCallback
  {
  protected:
    LexerCallback() {}
    LexerCallback(const LexerCallback&) {}
    ~LexerCallback() {}
    LexerCallback& operator =(const LexerCallback&) { return *this; }
  public:
    virtual bool comment(Stream& stream, Token t) = 0;
    virtual bool directive(Stream& stream, Token t) = 0;
    virtual bool identifier(Stream& stream, Token t) = 0;
    virtual bool literal(Stream& stream, Token t) = 0;
    virtual bool op(Stream& stream, OPERATOR op, Token t) = 0;
    virtual bool eol(Stream& stream) = 0;
    virtual void eof(Stream& stream) = 0;
  };

  class Stream
  {
  private:
    Stream(const Stream&);
    Stream& operator =(const Stream&);

    const char* ptr;
    const char* end;

    const char* last_break;
    int line_number;

    const char* stream_name;

    Log& log;

  public:
    Stream(const char* begin, const char* end, const char* name, Log& log);

    bool eof() const { return ptr >= end; }

    const char* current() const
    {
      return ptr;
    }
    
    const char* get()
    {
      if (*ptr == '\n')
      {
        last_break = ptr;
        ++line_number;
      }
      return ptr++;
    }

    void warning(const char* message);
    void warning(const std::string& message);
    void error(const char* message);
    void error(const std::string& message);
  };

  Stream& consume(Stream& stream, LexerCallback& callback);
}

#endif  // INCLUDED_GLSLSTREAM_H
