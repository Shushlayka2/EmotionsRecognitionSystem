#include "Chameleon.h"

Chameleon::Chameleon(const int i) {
    value = std::to_string(i);
}

Chameleon::Chameleon(const float f) {
    value = std::to_string(f);
}

Chameleon::Chameleon(std::string const& s) {
    value = s;
}

Chameleon::Chameleon(Chameleon const& other) {
    value = other.value;
}

Chameleon& Chameleon::operator=(int i) {
    value = std::to_string(i);
    return *this;
}

Chameleon& Chameleon::operator=(float f) {
    value = std::to_string(f);
    return *this;
}

Chameleon& Chameleon::operator=(std::string const& s) {
    value = s;
    return *this;
}

Chameleon& Chameleon::operator=(Chameleon const& other) {
    value = other.value;
    return *this;
}

Chameleon::operator int() const {
    return std::stoi(value.c_str());
}

Chameleon::operator float() const {
    return std::stof(value.c_str());
}

Chameleon::operator std::string() const {
    return value;
}