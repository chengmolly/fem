#pragma once

#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace fem
{

    class Timer
    {
    public:
        using clock = std::chrono::high_resolution_clock;
        using time_point = std::chrono::time_point<clock>;
        using duration = std::chrono::duration<double>;

        Timer() : start_time_(clock::now()), is_running_(true) {}

        void start()
        {
            start_time_ = clock::now();
            is_running_ = true;
        }

        void stop()
        {
            if (is_running_)
            {
                elapsed_ += clock::now() - start_time_;
                is_running_ = false;
            }
        }

        void reset()
        {
            elapsed_ = duration::zero();
            is_running_ = true;
            start_time_ = clock::now();
        }

        double elapsed() const
        {
            if (is_running_)
            {
                return std::chrono::duration<double>(clock::now() - start_time_).count() + elapsed_.count();
            }
            return elapsed_.count();
        }

        double elapsed_ms() const { return elapsed() * 1000.0; }

        double elapsed_us() const { return elapsed() * 1e6; }

        explicit operator double() const { return elapsed(); }

    private:
        time_point start_time_;
        duration elapsed_{duration::zero()};
        bool is_running_;
    };

    class TimerManager
    {
    public:
        static TimerManager &instance()
        {
            static TimerManager instance;
            return instance;
        }

        void start(const std::string &name)
        {
            auto &timer = timers_[name];
            timer.start();
        }

        void stop(const std::string &name)
        {
            timers_[name].stop();
        }

        double get(const std::string &name) const
        {
            auto it = timers_.find(name);
            return (it != timers_.end()) ? it->second.elapsed() : 0.0;
        }

        void reset(const std::string &name)
        {
            if (timers_.find(name) != timers_.end())
            {
                timers_[name].reset();
            }
        }

        void reset_all()
        {
            for (auto &[_, timer] : timers_)
            {
                timer.reset();
            }
        }

        void print_summary() const
        {
            std::cout << "\n=== Timer Summary ===\n";
            std::cout << std::left;

            std::vector<std::pair<std::string, double>> sorted;
            for (const auto &[name, timer] : timers_)
            {
                sorted.emplace_back(name, timer.elapsed());
            }

            std::sort(sorted.begin(), sorted.end(),
                      [](const auto &a, const auto &b)
                      {
                          return a.second > b.second;
                      });

            double total = 0.0;
            for (const auto &[name, time] : sorted)
            {
                total += time;
            }

            for (const auto &[name, time] : sorted)
            {
                std::cout << std::setw(30) << name
                          << std::setw(12) << std::fixed << std::setprecision(3)
                          << time << "s";
                if (total > 0)
                {
                    std::cout << "(" << std::setprecision(1)
                              << (time / total * 100) << "%)";
                }
                std::cout << "\n";
            }

            std::cout << std::setw(30) << "Total"
                      << std::setw(12) << total << "s\n";
        }

        void print_header(const std::string &title) const
        {
            std::cout << "\n"
                      << title << "\n";
            std::cout << std::string(title.length(), '=') << "\n";
        }

        class ScopedTimer
        {
        public:
            ScopedTimer(const std::string &name) : name_(name)
            {
                TimerManager::instance().start(name);
            }

            ~ScopedTimer()
            {
                TimerManager::instance().stop(name_);
            }

        private:
            std::string name_;
        };

#define FEM_SCOPED_TIMER(name) \
    fem::TimerManager::ScopedTimer _timer_##name(#name);

    private:
        TimerManager() = default;
        std::map<std::string, Timer> timers_;
    };

    inline Timer &ger_timer(const std::string &name)
    {
        static std::map<std::string, Timer> timers;
        return timers[name];
    }
} // namespace fem