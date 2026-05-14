#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>
#include <stdexcept>

namespace fem
{
    template <size_t M = 0, size_t N = 0, typename T>
    class DenseMatrix
    {
    public:
        using value_type = T;
        using size_type = size_t;
        static constexpr size_type rows_at_compile_time = M;
        static constexpr size_type cols_at_compile_time = N;

        DenseMatrix() : rows_(0), cols_(0), data_() {}
        explicit DenseMatrix(size_type rows, size_type cols) : rows_(rows), cols_(cols), data_(rows * cols) {}
        DenseMatrix(size_type rows, size_type cols, const T &init_val) : rows_(rows), cols_(cols), data_(rows * cols, init_val) {}
        DenseMatrix(std::initializer_list<std::initializer_list<T>> init)
            : rows_(init.size()),
              cols_(init.begin()->size()),
              data_()
        {
            data_.reserve(rows_ * cols_);
            for (const auto &row : init)
            {
                data_.insert(data_.end(), row.begin(), row.end());
            }
        }

        size_type rows() const { return rows_; }
        size_type cols() const { return cols_; }
        size_type size() const { return rows_ * cols_; }
        bool empty() const { return data_.empty(); }

        void resize(size_type rows, size_type cols)
        {
            rows_ = rows;
            cols_ = cols;
            data_.resize(rows_ * cols_);
        }

        T &operator()(size_type i, size_type j)
        {
            return data_[i * cols_ + j];
        }

        const T &operator()(size_type i, size_type j) const
        {
            return data_[i * cols_ + j];
        }

        T &at(size_type i, size_type j)
        {
            if (i >= rows_ || j >= cols_)
            {
                throw std::out_of_range("index out of range");
            }
            return (*this)(i, j);
        }

        const T &at(size_type i, size_type j) const
        {
            if (i >= rows_ || j >= cols_)
            {
                throw std::out_of_range("index out of range");
            }
            return (*this)(i, j);
        }

        std::vector<T> get_row(size_type i) const
        {
            if (i >= rows_)
            {
                throw std::out_of_range("index out of range");
            }
            std::vector<T> row(cols_);
            std::copy(data_.begin() + i * cols_, data_.begin() + (i + 1) * cols_, row.begin());
            return row;
        }

        std::vector<T> get_col(size_type j) const
        {
            if (j >= cols_)
            {
                throw std::out_of_range("index out of range");
            }
            std::vector<T> col(rows_);
            for (size_type i = 0; i < rows_; ++i)
            {
                col[i] = (*this)(i, j);
            }
            return col;
        }

        void set_row(size_type i, const std::vector<T> &row)
        {
            if (i >= rows_ || row.size() != cols_)
            {
                throw std::invalid_argument("Row size mismatch");
            }
            std::copy(row.begin(), row.end(), data_.begin() + i * cols_);
        }

        void set_col(size_type j, const std::vector<T> &col)
        {
            if (j >= cols_ || col.size() != rows_)
            {
                throw std::invalid_argument("Column size mismatch");
            }
            for (size_type i = 0; i < rows_; ++i)
            {
                (*this)(i, j) = col[i];
            }
        }

        T *data() { return data_.data(); }
        const T *data() const { return data_.data(); }

        void fill(const T &val)
        {
            return std::fill(data_.begin(), data_.end(), val);
        }

        void zero()
        {
            fill(T{});
        }

        void identity(size_type n = 0)
        {
            size_type dim = (n == 0) ? std::min(rows_, cols_) : n;
            fill(T{});
            for (size_type i = 0; i < dim; ++i)
            {
                (*this)(i, i) = 1;
            }
        }

        DenseMatrix operator+(const DenseMatrix &other) const
        {
            if (rows_ != other.rows_ || cols_ != other.cols_)
            {
                throw std::invalid_argument("Matrix size mismatch");
            }
            DenseMatrix res(rows_, cols_);
            for (size_type i = 0; i < size(); ++i)
            {
                res.data_[i] = data_[i] + other.data_[i];
            }
            return res;
        }

        DenseMatrix operator-(const DenseMatrix &other) const
        {
            if (rows_ != other.rows_ || cols_ != other.cols_)
            {
                throw std::invalid_argument("Matrix size mismatch");
            }
            DenseMatrix res(rows_, cols_);
            for (size_type i = 0; i < size(); ++i)
            {
                res.data_[i] = data_[i] - other.data_[i];
            }
            return res;
        }

        DenseMatrix operator*(const T &scala) const
        {
            DenseMatrix res(rows_, cols_);
            for (size_type i; i < size(); ++i)
            {
                res.data_[i] = data_[i] * scala;
            }
            return res;
        }

        std::vector<T> operator*(const std::vector<T> &vec) const
        {
            if (vec.size() != cols_)
            {
                throw std::invalid_argument("Matrix-Vector size mismatch");
            }
            std::vector<T> res(rows_, T{});
            for (size_type i = 0; i < rows_; ++i)
            {
                T sum = 0;
                for (size_type j = 0; j < cols_; ++j)
                {
                    sum += (*this)(i, j) * vec[j];
                }
                res[i] = sum;
            }
            return res;
        }

        DenseMatrix operator*(const DenseMatrix &other) const
        {
            if (cols_ != other.rows_)
            {
                throw std::invalid_argument("Matrix-Matrix size mismatch");
            }
            DenseMatrix res(rows_, other.cols_, T{});
            for (size_type i = 0; i < rows_; ++i)
            {
                for (size_type j = 0; j < other.cols_; ++j)
                {
                    T sum = 0;
                    for (size_type k = 0; k < cols_; ++k)
                    {
                        sum += (*this)(i, k) * other(k, j);
                    }
                    res(i, j) = sum;
                }
            }
            return res;
        }

        DenseMatrix transpose() const
        {
            DenseMatrix res(rows_, cols_);
            for (size_type i = 0; i < rows_; ++i)
            {
                for (size_type j = 0; j < cols_; ++j)
                {
                    res(i, j) = (*this)(j, i);
                }
            }
            return res;
        }

        DenseMatrix block(size_type row_start, size_type col_start, size_type row_count, size_type col_count) const
        {
            DenseMatrix res(row_count, col_count);
            for (size_type i = 0; i < row_count; ++i)
            {
                for (size_type j = 0; j < col_count; ++j)
                {
                    res(i, j) = (*this)(row_start + i, col_start + j);
                }
            }
            return res;
        }

        T trace() const
        {
            size_type min_dim = std::min(rows_, cols_);
            T sum = 0;
            for (size_type i = 0; i < min_dim; ++i)
            {
                sum += (*this)(i, i);
            }
            return sum;
        }

        T norm_frobenius() const
        {
            T sum = 0;
            for (const auto &val : data_)
            {
                sum += val * val;
            }
            return std::sqrt(sum);
        }

        T norm_l1() const
        {
            T max_col_sum = 0;
            for (size_type j = 0; j < cols_; ++j)
            {
                T col_sum = 0;
                for (size_type i = 0; i < rows_; ++i)
                {
                    col_sum += std::abs((*this)(i, j));
                }
                max_col_sum = std::max(max_col_sum, col_sum);
            }
            return max_col_sum;
        }

        T norm_inf() const
        {
            T max_row_sum = 0;
            for (size_type i = 0; i < rows_; ++i)
            {
                T row_sum = 0;
                for (size_type j = 0; j < cols_; ++j)
                {
                    row_sum += std::abs((*this)(i, j));
                }
                max_row_sum = std::max(max_row_sum, row_sum);
            }
            return max_row_sum;
        }

        std::vector<T> diag() const
        {
            size_type min_dim = std::min(rows_, cols_);
            std::vector<T> diag(min_dim);
            for (size_type i = 0; i < min_dim; ++i)
            {
                diag[i] = (*this)(i, i);
            }
            return diag;
        }

        void print() const
        {
            for (size_type i = 0; i < rows_; ++i)
            {
                for (size_type j = 0; j < cols_; ++j)
                {
                    std::cout << (*this)(i, j) << "\t";
                }
                std::cout << "\n";
            }
        }

    private:
        size_type rows_;
        size_type cols_;
        std::vector<T> data_;
    };
}