#pragma once

#include <string>

class Chameleon {
public:
    Chameleon() {};
    Chameleon(const int);
    Chameleon(const float);
    Chameleon(const std::string&);
    Chameleon(const Chameleon&);

    Chameleon& operator=(int);
    Chameleon& operator=(float);
    Chameleon& operator=(std::string const&);
    Chameleon& operator=(Chameleon const&);

    operator int() const;
    operator float() const;
    operator std::string() const;
  
private:
    std::string value;
};