#pragma once

#include <string>
#include <map>

#include "Chameleon.h"

class ConfigHandler {
    std::map<std::string, Chameleon> content;

public:
    ConfigHandler();

    ConfigHandler(std::string const& configFile);

    Chameleon const& Value(std::string const& entry) const;
};