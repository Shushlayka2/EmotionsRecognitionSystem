#pragma once

#include <string>
#include <vector>
#include <map>

#include "Chameleon.h"

class ConfigHandler {
private:
    std::map<std::string, Chameleon> content;
    std::vector<int> split(const std::string& s) const;
public:
    ConfigHandler();
    ConfigHandler(std::string const& configFile);
    Chameleon const& Value(std::string const& entry) const;
    std::vector<int> VectorValue(std::string const& entry) const;
};