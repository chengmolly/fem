#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <fstream>

namespace fem
{

    enum class LogLevel
    {
        Debug,
        Info,
        Waring,
        Error,
        Fatal
    };

    class LogSink
    {
    public:
        virtual ~LogSink() = default;
        virtual void write(LogLevel level, const std::string &msg) = 0;
    };

    class ConsoleSink : public LogSink
    {
    public:
        explicit ConsoleSink(bool use_color = true) : use_color_(use_color) {}
        void write(LogLevel level, const std::string &msg) override
        {
            if (use_color_)
            {
                std::cout << color_code(level);
            }
            std::cout << "[" << level_string(level) << "]" << msg;
            if (use_color_)
            {
                std::cout << "\033[0m";
            }
            std::cout << "\n";
        }

    private:
        static const char *color_code(LogLevel level)
        {
            switch (level)
            {
            case LogLevel::Debug:
                return "\033[36m";
            case LogLevel::Info:
                return "\033[32m";
            case LogLevel::Waring:
                return "\033[33m";
            case LogLevel::Error:
                return "\033[31m";
            case LogLevel::Fatal:
                return "\033[35m";
            default:
                return "\033[0m";
            }
        }

        static const char *level_string(LogLevel level)
        {
            switch (level)
            {
            case LogLevel::Debug:
                return "DEBUG";
            case LogLevel::Info:
                return "INFO";
            case LogLevel::Waring:
                return "WARN";
            case LogLevel::Error:
                return "ERROR";
            case LogLevel::Fatal:
                return "FATAL";
            default:
                return "UNKNOWN";
            }
        }

        bool use_color_;
    };

    class FileSink : public LogSink
    {
    public:
        explicit FileSink(const std::string &filename) : file_(filename, std::ios::app)
        {
            if (!file_.is_open())
            {
                throw std::runtime_error("Cannot open log file: " + filename);
            }
        }

        ~FileSink() override
        {
            if (file_.is_open())
                file_.close();
        }

        void write(LogLevel level, const std::string &msg) override
        {
            auto now = std::chrono::system_clock::now();
            auto time = std::chrono::system_clock::to_time_t(now);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

            std::ostringstream oss;
            oss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
            oss << '.' << std::setfill('0') << std::setw(3) << ms.count();

            file_ << "[" << oss.str() << "]"
                  << "[" << level_string(level) << "]"
                  << msg << "\n";
            file_.flush();
        }

    private:
        static const char *level_string(LogLevel level)
        {
            switch (level)
            {
            case LogLevel::Debug:
                return "DEBUG";
            case LogLevel::Info:
                return "INFO";
            case LogLevel::Waring:
                return "WARN";
            case LogLevel::Error:
                return "ERROR";
            case LogLevel::Fatal:
                return "FATAL";
            default:
                return "UNKNOWN";
            }
        }

        std::ofstream file_;
    };

    class Logger
    {
    public:
        static Logger &instance()
        {
            static Logger instance;
            return instance;
        }

        void add_sink(std::shared_ptr<LogSink> sink)
        {
            sinks_.push_back(sink);
        }

        void set_level(LogLevel level)
        {
            min_level_ = level;
        }

        template <typename... Args>
        void info(const std::string &fmt, Args &&...args)
        {
            log(LogLevel::Info, format(fmt, std::forward<Args>(args)...));
        }

        template <typename... Args>
        void debug(const std::string &fmt, Args &&...args)
        {
            log(LogLevel::Debug, format(fmt, std::forward<Args>(args)...));
        }

        template <typename... Args>
        void error(const std::string &fmt, Args &&...args)
        {
            log(LogLevel::Error, format(fmt, std::forward<Args>(args)...));
        }

        template <typename... Args>
        void warning(const std::string &fmt, Args &&...args)
        {
            log(LogLevel::Waring, format(fmt, std::forward<Args>(args)...));
        }
        template <typename... Args>
        void fatal(const std::string &fmt, Args &&...args)
        {
            log(LogLevel::Fatal, format(fmt, std::forward<Args>(args)...));
        }

    private:
        Logger()
        {
            add_sink(std::make_shared<ConsoleSink>());
        }

        void log(LogLevel level, const std::string &msg)
        {
            if (level < min_level_)
                return;
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto &sink : sinks_)
            {
                sink->write(level, msg);
            }
        }

        template <typename T>
        static std::string format_arg(const T &arg)
        {
            std::ostringstream oss;
            oss << arg;
            return oss.str();
        }

        template <typename... Args>
        static std::string format(const std::string &fmt, Args &&...args)
        {
            std::vector<std::string> args_str = {format_arg(args)...};
            size_t arg_idx = 0;

            std::ostringstream result;
            size_t pos = 0;
            while (pos < fmt.size() && arg_idx < args_str.size())
            {
                if (fmt[pos] == '{' && pos + 1 < fmt.size() && fmt[pos + 1] == '}')
                {
                    result << args_str[arg_idx++];
                    pos += 2;
                }
                else
                {
                    result << fmt[pos++];
                }
            }
            while (pos < fmt.size())
            {
                result << fmt[pos++];
            }
            return result.str();
        }

        std::vector<std::shared_ptr<LogSink>> sinks_;
        LogLevel min_level_ = LogLevel::Info;
        std::mutex mutex_;
    };

#define LOG_INFO(fmt, ...) fem::Logger::instance().info(fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) fem::Logger::instance().debug(fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) fem::Logger::instance().error(fmt, ##__VA_ARGS__)
#define LOG_WARNING(fmt, ...) fem::Logger::instance().warning(fmt, ##__VA_ARGS__)
#define LOG_FATAL(fmt, ...) fem::Logger::instance().fatal(fmt, ##__VA_ARGS__)
} // namespace fem