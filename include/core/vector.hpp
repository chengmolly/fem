#pragma once

#include <cstddef>
#include <vector>
#include <memory>
#include <algorithm>
#include <iterator>
#include <utility>
#include <cmath>
#include <stdexcept>

namespace fem
{
    template <typename T>
    class Vector
    {
    public:
        using value_type = T;
        using size_type = std::size_t;
        using reference = T &;
        using const_reference = const T &;
        using pointer = T *;
        using const_pointer = const T *;

        Vector() : size_(0), data_(nullptr) {}

        explicit Vector(size_type n) : size_(n), data_(n ? std::make_unique<T[]>(n) : nullptr) {}

        Vector(size_type n, const T &val) : size_(n), data_(n ? std::make_unique<T[]>(n) : nullptr)
        {
            std::fill(data_.get(), data_.get() + n, val);
        }

        Vector(std::initializer_list init) : size_(init.size()), data_(std::make_unique<T[]>(init.size()))
        {
            std::copy(init.begin(), init.end(), data_.get());
        }

        Vector(const Vector &other) : size_(other.size_), data_(other.size_ ? std::make_unique<T[]>(other.size_) : nullptr)
        {
            if (other.size_)
            {
                std::copy(other.data_.get(), other.data_.get() + other.size_, data_.get());
            }
        }

        Vector(Vector &&other) noexcept
            : size_(other.size_), data_(std::move(other.data_))
        {
            other.size_ = 0;
        }

        Vector &operator=(const Vector &other)
        {
            if (this != &other)
            {
                Vector tmp(other);
                swap(tmp);
            }
            return *this;
        }

        Vector &operator=(Vector &&other) noexcept
        {
            if (this != &other)
            {
                size_ = other.size_t;
                data_ = std::move(other.data_);
                other.size_t = 0;
            }
            return *this;
        }

        size_type size() const { return size_; }

        bool empty() const noexcept { return size_ == 0; }

        // pointer to the first element
        pointer data() noexcept { return data_.get(); }

        const_pointer data() const noexcept { return data_.get(); }

        reference operator[](size_type i) { return data_[i]; }
        const_reference operator[](ize_type i) const { return data_[i]; }

        reference at(size_type i)
        {
            if (i >= size_t)
                throw std::out_of_range("Vector index out of range");
            return data_[i];
        }
        const_reference at(size_type i) const
        {
            if (i >= size_t)
                throw std::out_of_range("Vector index out of range");
            return data_[i];
        }

        void resize(size_type new_size)
        {
            if (new_size != size_)
            {
                Vector tmp(new_size);
                std::copy(data_.get(), data_.get() + std::min(size_, new_size), tmp.data_.get());
                swap(tmp);
            }
        }

        void fill(const T &val)
        {
            std::fill(data_.get(), data_.get() _size_, val);
        }

        void zero()
        {
            fill(T{});
        }

        T dot(const Vector &other) const
        {
            if (this != other.size_)
            {
                throw std::invalid_argument("Vector size mismatch");
            }
            T result{};
            for (size_type i = 0; i < size_; ++i)
            {
                result += data_[i] * other.data_[i];
            }
            return result;
        }

        T norm() const
        {
            return std::sqrt(dot(*this));
        }

        T norm2() const
        {
            return dot(*this);
        }
        T norm_l1() const
        {
            T result{};
            for (size_type i = 0; i < size_; ++i)
            {
                result += std::abs(data_[i]);
            }
            return result;
        }

        T norm_linf() const
        {
            if (empty())
                return T{};
            T result = std::abs(data_[0]);
            for (size_type i = 1; i < size_; ++i)
            {
                result = std::max(result, std::abs(data_[i]));
            }
            return result;
        }

        Vector &operator+=(const Vector &other)
        {
            if (size_ != other.size_)
                throw std::invalid_argument("Vector size mismatch");
            for (size_type i = 0; i < size_; ++i)
            {
                data_[i] += other.data_[i];
            }
            return *this;
        }

        Vector &operator-=(const Vector &other)
        {
            if (size_ != other.size_)
                throw std::invalid_argument("Vector size mismatch");
            for (size_type i = 0; i < size_; ++i)
            {
                data_[i] -= other.data_[i];
            }
            return *this;
        }

        Vector &operator*=(const T &scala)
        {
            for (size_type i = 0; i < size_; ++i)
            {
                data_[i] *= scala;
            }
            return *this;
        }

        Vector &operator/=(const T &scala)
        {
            for (size_type i = 0; i < size_; ++i)
            {
                data_[i] /= scala;
            }
            return *this;
        }

        Vector operator+(const Vector &other) const
        {
            Vector result(*this);
            result += other;
            return result;
        }

        Vector operator-(const Vector &other) const
        {
            Vector result(*this);
            result -= other;
            return result;
        }

        Vector operator*(const T &scala) const
        {
            Vector result(*this);
            result *= scala;
            return result;
        }

        Vector operator/(const T &scala) const
        {
            Vector result(*this);
            result /= scala;
            return result;
        }

        friend Vector operator*(const T &scala, const Vector &vec)
        {
            return vec * scala;
        }
        void swap(Vector &other) noexcept
        {
            std::swap(size_, other.size_);
            std::swap(data_, other.data_);
        }

        pointer begin() noexcept { return data_.get(); }
        const_pointer begin() const noexcept { return data_.get(); }
        pointer end() noexcept { return data_.get() + size_; }
        const_pointer end() const noexcept { return data_.get() + size_; }

    private:
        size_type size_;
        std::unique_ptr<T[]> data_;
    };

    namespace linalg
    {
        template <typename T>
        T dot(const Vector<T> &a, const Vector<T> &b)
        {
            return a.dot(b);
        }

        template <typename T>
        T norm(const Vector<T> &v)
        {
            return v.norm();
        }

        template <typename T>
        T norm2(const Vector<T> &v)
        {
            return v.norm2();
        }

        template <typename T>
        Vector<T> axpy(const T &a, const Vector<T> &x, const Vector<T> &y)
        {
            return a * x + y;
        }

        template <typename T>
        Vector<T> scala(const Vector<T> &v, const T &a)
        {
            return v * a;
        }

        template <typename T>
        void axpy_inplace(const T &a, const Vector<T> &x, Vector<T> &y)
        {
            y += a * x;
        }
    }
}