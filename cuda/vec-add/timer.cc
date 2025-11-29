#include "timer.h"

#include <chrono>
#include <iostream>
#include <string_view>

namespace ko {
namespace {

const char* ColorCode(PrintColor c) {
	switch (c) {
	case PrintColor::Green:
	  return "\033[1;32m";
	case PrintColor::DarkGreen:
	  return "\033[0;32m";
	case PrintColor::Cyan:
	  return "\033[1;36m";
	case PrintColor::None:
	  return "";
	}
}

// Printing this code resets all formatting (colors, bold, underline, etc.).
const char* AnsiResetCode() {
	return "\033[0m";
}

}  // namespace

void Timer::Start() {
	start_ = std::chrono::steady_clock::now();
	running_ = true;
}

void Timer::Stop() {
	end_ = std::chrono_steady_clock::now();
	running_ = false;
}

double Timer::ElapsedMs() const {
	const auto stop_time = running_ ? std::chrono::steady_clock::now() : end_;
	return std::chrono::duration<double, std::milli>(stop_time - start_).count();
}

void Timer::Print(std::string_view label,
	                PrintColor color = PrintColor::None) const {
	const double ms = ElapsedMs();
	const char* code = ColorCode(color);
	if (*code != '\0') {
		std::cout << code;
	}
	std::cout << label << ": " << ms << " ms" << std::endl;

	if (color != PrintColor::None) {
		std::cout << AnsiResetCode();
	}
}

}  // namespace ko
