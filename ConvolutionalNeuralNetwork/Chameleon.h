#pragma once

#include <string>

class Chameleon {
private:
    std::string value;
public:
    Chameleon() {};
    Chameleon(const int i);
    Chameleon(const float f);
    Chameleon(const std::string& s);
    Chameleon(const Chameleon& other);

    Chameleon& operator=(int i);
    Chameleon& operator=(float f);
    Chameleon& operator=(std::string const& s);
    Chameleon& operator=(Chameleon const& other);

    operator int() const;
    operator float() const;
    operator std::string() const;
};