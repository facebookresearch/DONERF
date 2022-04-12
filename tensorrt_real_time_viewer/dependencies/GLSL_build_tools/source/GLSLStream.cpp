


#include <string>
#include <stdexcept>

#include "GLSLStream.h"


namespace
{
  class lexer_error : public std::runtime_error
  {
  public:
    lexer_error(const std::string& msg)
      : runtime_error(msg)
    {
    }
  };

  void invalidCharacter(GLSL::Stream& stream)
  {
    std::string msg = std::string("invalid input character: '") + *stream.current() + '\'';
    stream.error(msg.c_str());
    throw lexer_error(msg);
  }


  bool isalpha(char c)
  {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '$' || c == '_' || c == '.' || c == '@';
  }

  bool isdigit_dec(char c)
  {
    return c >= '0' && c <= '9';
  }

  bool isdigit_oct(char c)
  {
    return c >= '0' && c <= '7';
  }

  bool isdigit_hex(char c)
  {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
  }

  bool isdigit_bin(char c)
  {
    return c == '0' || c == '1';
  }

  bool isalnum(char c)
  {
    return isalpha(c) || isdigit_dec(c);
  }

  GLSL::Stream& next(GLSL::Stream& stream)
  {
    stream.get();
    if (stream.eof())
    {
      const char* msg = "unexpected end of file";
      stream.error(msg);
      throw lexer_error(msg);
    }
    return stream;
  }

  template <typename F>
  GLSL::Stream& readSequence(GLSL::Stream& stream, F f)
  {
    while (!stream.eof() && f(*stream.current()))
      stream.get();
    return stream;
  }

  GLSL::Stream& readLineComment(GLSL::Stream& stream)
  {
    return readSequence(stream, [](char c) { return c != '\n'; });
  }

  GLSL::Stream& readBlockComment(GLSL::Stream& stream)
  {
    char c0 = *next(stream).current();
    while (true)
    {
      char c1 = *next(stream).current();
      if (c0 == '*' && c1 == '/')
      {
        break;
      }
      c0 = c1;
    }
    return stream;
  }

  GLSL::Stream& readIdentifier(GLSL::Stream& stream)
  {
    return readSequence(stream, isalnum);
  }

  GLSL::Stream& readStringLiteral(GLSL::Stream& stream)
  {
    bool escape = false;
    while (true)
    {
      next(stream);
      if (*stream.current() == '\\' && !escape)
      {
        escape = true;
        continue;
      }

      if (*stream.current() == '"')
      {
        stream.get();
        break;
      }
      else if (*stream.current() == '\n')
      {
        stream.error("line break in string literal");
        throw lexer_error("line break in string literal");
      }

      escape = false;
    }
    return stream;
  }

  GLSL::Stream& readFloatLiteralExponent(GLSL::Stream& stream)
  {
    if (!stream.eof() && (*stream.current() == 'e' || *stream.current() == 'E'))
    {
      next(stream);
      if (*stream.current() == '+' || *stream.current() == '-')
        next(stream);
      if (!isdigit_dec(*stream.current()))
        invalidCharacter(stream);
      return readSequence(stream, isdigit_dec);
    }
    return stream;
  }

  GLSL::Stream& readFloatLiteralFraction(GLSL::Stream& stream)
  {
    return readFloatLiteralExponent(readSequence(stream, isdigit_dec));
  }

  GLSL::Stream& readIntegerLiteralSuffix(GLSL::Stream& stream)
  {
    if (!stream.eof() && *stream.current() == 'U')
      stream.get();
    return stream;
  }

  GLSL::Stream& readNumberLiteral(GLSL::Stream& stream)
  {
    readSequence(stream, isdigit_dec);

    if (!stream.eof())
    {
      if (*stream.current() == '.')
      {
        stream.get();
        return readFloatLiteralFraction(stream);
      }
      return readIntegerLiteralSuffix(stream);
    }

    return stream;
  }

  GLSL::Stream& readNumberLiteralPrefix(GLSL::Stream& stream)
  {
    const char* begin = stream.current();
    if (*stream.current() == 'x' || *stream.current() == 'X')
    {
      if (readIntegerLiteralSuffix(readSequence(stream, isdigit_hex)).current() - begin < 1)
        invalidCharacter(stream);
    }
    else if (*stream.current() == 'b' || *stream.current() == 'B')
    {
      if (readIntegerLiteralSuffix(readSequence(stream, isdigit_bin)).current() - begin < 1)
        invalidCharacter(stream);
    }
    else if (*stream.current() == 'f' || *stream.current() == 'F')
    {
      if (readSequence(stream, isdigit_hex).current() - begin < 1)
        invalidCharacter(stream);
    }
    else if (*stream.current() == 'd' || *stream.current() == 'D')
    {
      if (readSequence(stream, isdigit_hex).current() - begin < 1)
        invalidCharacter(stream);
    }
    else if (isdigit_dec(*stream.current()))
      return readNumberLiteral(stream);
    return stream;
  }

  GLSL::OPERATOR getDualCharacterOp(GLSL::Stream& stream, GLSL::OPERATOR op1, char c2, GLSL::OPERATOR op2)
  {
    if (!stream.eof() && (*stream.current() == c2))
    {
      stream.get();
      return op2;
    }
    return op1;
  }

  GLSL::OPERATOR getDualCharacterOp(GLSL::Stream& stream, GLSL::OPERATOR op1, char c2, GLSL::OPERATOR op2, char c3, GLSL::OPERATOR op3)
  {
    if (!stream.eof())
    {
      if (*stream.current() == c2)
      {
        stream.get();
        return op2;
      }
      if (*stream.current() == c3)
      {
        stream.get();
        return op3;
      }
    }
    return op1;
  }

  GLSL::OPERATOR getOperator(GLSL::Stream& stream)
  {
    switch(*stream.get())
    {
      case '+':
        return GLSL::OP_PLUS;
      case '-':
        return GLSL::OP_MINUS;
      case '*':
        return GLSL::OP_ASTERISK;
      case '^':
        return GLSL::OP_CIRCUMFLEX;
      case '~':
        return GLSL::OP_TILDE;
      case '(':
        return GLSL::OP_LPARENT;
      case ')':
        return GLSL::OP_RPARENT;
      case '{':
        return GLSL::OP_LBRACE;
      case '}':
        return GLSL::OP_RBRACE;
      case '[':
        return GLSL::OP_LBRACKET;
      case ']':
        return GLSL::OP_RBRACKET;
      case '?':
        return GLSL::OP_QUEST;
      case ':':
        return GLSL::OP_COLON;
      case ',':
        return GLSL::OP_COMMA;
      case ';':
        return GLSL::OP_SEMICOLON;
      case '<':
        return getDualCharacterOp(stream, GLSL::OP_LT, '=', GLSL::OP_LEQ, '<', GLSL::OP_LL);
      case '>':
        return getDualCharacterOp(stream, GLSL::OP_GT, '=', GLSL::OP_GEQ, '<', GLSL::OP_GG);
      case '&':
        return getDualCharacterOp(stream, GLSL::OP_AND, '&', GLSL::OP_AAND);
      case '|':
        return getDualCharacterOp(stream, GLSL::OP_OR, '&', GLSL::OP_OOR);
      case '=':
        return getDualCharacterOp(stream, GLSL::OP_EQ, '=', GLSL::OP_EEQ);
      case '!':
        return getDualCharacterOp(stream, GLSL::OP_BANG, '=', GLSL::OP_NEQ);
    }

    invalidCharacter(stream);
    return GLSL::OP_INVALID;
  }
}

namespace GLSL
{
  Stream::Stream(const char* begin, const char* end, const char* name, Log& log)
    : ptr(begin),
      end(end),
      last_break(nullptr),
      line_number(1),
      stream_name(name),
      log(log)
  {
  }

  Stream& consume(Stream& stream, LexerCallback& callback)
  {
    while (!stream.eof())
    {
      switch (*stream.current())
      {
        case '/':
        {
          const char* begin = stream.get();

          if (!stream.eof())
          {
            if (*stream.current() == '/')
            {
              readLineComment(stream);
              if (!callback.comment(stream, Token(begin, stream.current())))
                return stream;
              break;
            }
            else if (*stream.current() == '*')
            {
              readBlockComment(stream);
              if (!callback.comment(stream, Token(begin, stream.current())))
                return stream;
              break;
            }
          }

          if (!callback.op(stream, OP_SLASH, Token(begin, stream.current())))
            return stream;
          break;
        }

        case '#':
        {
          const char* begin = stream.get();

          if (!stream.eof())
          {
            if (isalpha(*stream.current()))
            {
              readIdentifier(stream);
              if (!callback.directive(stream, Token(begin, stream.current())))
                return stream;
              break;
            }
          }

          break;
        }

        case '.':
        {
          const char* begin = stream.get();

          if (!stream.eof())
          {
            //if (isalpha(*stream.current()))
            //{
            //  readIdentifier(stream);
            //  if (!callback.directive(stream, Token(begin, stream.current())))
            //    return stream;
            //  break;
            //}
            /*else */if (isdigit_dec(*stream.current()))
            {
              readFloatLiteralFraction(stream);
              if (!callback.literal(stream, Token(begin, stream.current())))
                return stream;
              break;
            }
          }

          if (!callback.op(stream, OP_DOT, Token(begin, stream.current())))
            return stream;
          break;
        }

        case '%':
        {
          const char* begin = stream.get();

          if (!stream.eof() && isalnum(*stream.current()))
          {
            readIdentifier(stream);
            if (!callback.identifier(stream, Token(begin, stream.current())))
              return stream;
            break;
          }

          if (!callback.op(stream, OP_PERCENT, Token(begin, stream.current())))
            return stream;
          break;
        }

        case '0':
        {
          const char* begin = stream.get();

          readNumberLiteralPrefix(stream);

          if (!callback.literal(stream, Token(begin, stream.current())))
            return stream;
          break;
        }

        case '"':
        {
          const char* begin = stream.get();
          readStringLiteral(stream);
          if (!callback.literal(stream, Token(begin, stream.current())))
              return stream;
          break;
        }

        case '\n':
          stream.get();
          if (!callback.eol(stream))
            return stream;
          break;
        case '\r':
        case '\t':
        case ' ':
          stream.get();
          break;

        default:
        {
          const char* begin = stream.current();

          if (isdigit_dec(*stream.current()))
          {
            readNumberLiteral(stream);
            if (!callback.literal(stream, Token(begin, stream.current())))
              return stream;
            break;
          }
          else if (isalnum(*stream.current()))
          {
            readIdentifier(stream);
            if (!callback.identifier(stream, Token(begin, stream.current())))
              return stream;
            break;
          }

          GLSL::OPERATOR op = getOperator(stream);

          if (!callback.op(stream, op, Token(begin, stream.current())))
            return stream;
          break;
        }
      }
    }
    callback.eol(stream);
    callback.eof(stream);
    return stream;
  }

  void Stream::warning(const char* message)
  {
    log.warning(message, stream_name, line_number);
  }

  void Stream::warning(const std::string& message)
  {
    log.warning(message, stream_name, line_number);
  }

  void Stream::error(const char* message)
  {
    log.error(message, stream_name, line_number);
  }

  void Stream::error(const std::string& message)
  {
    log.error(message, stream_name, line_number);
  }

  Token token(OPERATOR op)
  {
    switch (op)
    {
      case OP_PLUS:
        return Token("+", 1);
      case OP_MINUS:
        return Token("-", 1);
      case OP_ASTERISK:
        return Token("*", 1);
      case OP_SLASH:
        return Token("/", 1);
      case OP_CIRCUMFLEX:
        return Token("^", 1);
      case OP_TILDE:
        return Token("~", 1);
      case OP_LPARENT:
        return Token("(", 1);
      case OP_RPARENT:
        return Token(")", 1);
      case OP_LBRACKET:
        return Token("[", 1);
      case OP_RBRACKET:
        return Token("]", 1);
      case OP_LBRACE:
        return Token("{", 1);
      case OP_RBRACE:
        return Token("}", 1);
      case OP_QUEST:
        return Token("?", 1);
      case OP_DOT:
        return Token(".", 1);
      case OP_COLON:
        return Token(":", 1);
      case OP_COMMA:
        return Token(",", 1);
      case OP_SEMICOLON:
        return Token(";", 1);
      case OP_EQ:
        return Token("=", 1);
      case OP_EEQ:
        return Token("==", 2);
      case OP_NEQ:
        return Token("!=", 2);
      case OP_LT:
        return Token("<", 1);
      case OP_LEQ:
        return Token("<=", 2);
      case OP_LL:
        return Token("<<", 2);
      case OP_GT:
        return Token(">", 1);
      case OP_GEQ:
        return Token(">=", 2);
      case OP_GG:
        return Token(">>", 2);
      case OP_AND:
        return Token("&", 1);
      case OP_AAND:
        return Token("&&", 2);
      case OP_OR:
        return Token("|", 1);
      case OP_OOR:
        return Token("||", 2);
      case OP_BANG:
        return Token("!", 1);
      case OP_PERCENT:
        return Token("%", 1);
    }
    throw std::runtime_error("invalid enum");
  }
}
