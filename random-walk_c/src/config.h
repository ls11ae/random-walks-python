#pragma once

#define SAVE_IMAGE
#define PYTHON_PATH(file) "../../wrapper/python/data/" file
#define DATA_PATH(path) "../../data/" path

#ifdef _WIN32
static const char *CMD = R"(C:\Users\DevChris\AppData\Local\Programs\Python\Python313\python.exe D:\uni\random-walks\wrapper\python\tools\test.py)";
#elif __linux__
static const char *CMD = R"(/usr/bin/python3 /home/omar/CLionProjects/random-walks/wrapper/python/tools/test.py)";
#endif
