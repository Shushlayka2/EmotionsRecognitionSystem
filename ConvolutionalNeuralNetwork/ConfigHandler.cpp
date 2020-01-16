#include <fstream>

#include "CustomException.h"
#include "ConfigHandler.h"

std::string trim(std::string const& source, char const* delims = " \t\r\n") {
    std::string result(source);
    std::string::size_type index = result.find_last_not_of(delims);
    if (index != std::string::npos)
        result.erase(++index);

    index = result.find_first_not_of(delims);
    if (index != std::string::npos)
        result.erase(0, index);
    else
        result.erase();
    return result;
}

ConfigHandler::ConfigHandler(std::string const& configFile) {
    std::ifstream file(configFile);

    std::string line;
    std::string name;
    std::string value;
    int equal_pos;
    while (std::getline(file, line)) {

        if (!line.length()) continue;
        if (line[0] == '#') continue;
        if (line[0] == ';') continue;

        equal_pos = line.find('=');
        name = trim(line.substr(0, equal_pos));
        value = trim(line.substr(equal_pos + 1));

        content[name] = Chameleon(value);
    }
}

Chameleon const& ConfigHandler::Value(std::string const& entry) const {

    std::map<std::string, Chameleon>::const_iterator ci = content.find(entry);

    if (ci == content.end()) throw_line("does not exist");

    return ci->second;
}
