#ifndef _TIMER_H_
#define _TIMER_H_

#include <chrono>
#include <string_view>

namespace ko {

enum class PrintColor { None, Green, DarkGreen, Cyan };

class Timer {
public:
		Timer() = default;

		void Start();
		void Stop();

		// Returns elapsed time in milliseconds.
		double ElapsedMs() const;

		// Prints: "<label>: <time> ms"
		void Print(std::string_view label, PrintColor color = PrintColor::None) const;

private:
		clock::time_point start_{};
		clock::time_point end_{};
		bool running_ = false;
};

}  // namespace ko

#endif  // _TIMER_H_
