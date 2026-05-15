#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <utility>
#include <functional>
#include <unordered_map>

namespace fem
{
    enum class SparseFormat
    {
        CRS,
        CCS,
        ELL,
        COO,
        HYB
    };

    class SparseMatrixBase
    {
    public:
        virtual ~SparseMatrixBase() = default;
        virtual size_t rows() const = 0;
        virtual size_t cols() const = 0;
        virtual size_t vsize() const = 0;
        virtual double operator()(size_t i, size_t j) const = 0;
        virtual void set(size_t i, size_t j, double val) = 0;
        virtual void add(size_t i, size_t j, double val) = 0;
        virtual void clear() = 0;
        virtual void scale(double factor) = 0;
    };

    class SparseMatrixCRS : public SparseMatrixBase
    {
    public:
        SparseMatrixCRS() : rows_(0), cols_(0) {}
        SparseMatrixCRS(size_t rows, size_t cols, size_t vsize = 0)
            : rows_(rows), cols_(cols), row_ptr_(rows + 1, 0), col_idx_(vsize), values_(vsize) {}

        size_t rows() const override { return rows_; }
        size_t cols() const override { return cols_; }
        size_t vsize() const override { return values_.size(); }

        // build matrix from raw data
        void build(size_t m, size_t n, std::vector<size_t> &&row_offsets,
                   std::vector<size_t> &&col_indices, std::vector<double> &&vals)
        {
            rows_ = m;
            cols_ = n;
            row_ptr_ = std::move(row_offsets);
            col_idx_ = std::move(col_indices);
            values_ = std::move(vals);
        }

        // empty matrix
        void begin_assembly()
        {
            col_idx_.clear();
            values_.clear();
            std::fill(row_ptr_.begin(), row_ptr_.end(), 0);
        }

        // insert entry into matrix
        void insert_entry(int row, int col, double val)
        {
            col_idx_.push_back(col);
            values_.push_back(val);
            ++row_ptr_[row + 1];
        }

        void end_assembly()
        {
            for (size_t i = 1; i <= rows_; ++i)
            {
                row_ptr_[i] += row_ptr_[i - 1];
            }

            // sort by row
            std::vector<size_t> temp_ptr_ = row_ptr_;
            size_t nnz = col_idx_.size();
            std::vector<size_t> new_col_idx(nnz);
            std::vector<double> new_values(nnz);

            std::copy(col_idx_.begin(), col_idx_.end(), new_col_idx.begin());
            std::copy(values_.begin(), values_.end(), new_values.begin());

            for (size_t i = 0; i < rows_; ++i)
            {
                std::sort(new_col_idx.begin() + temp_ptr_[i], new_col_idx.begin() + temp_ptr_[i + 1]);
                std::sort(new_values.begin() + temp_ptr_[i], new_values.begin() + temp_ptr_[i + 1]);
            }
            col_idx_ = std::move(new_col_idx);
            values_ = std::move(new_values);
        }

        void compress()
        {
            if (col_idx_.empty())
                return;

            std::vector<size_t> new_row_ptr(rows_ + 1, 0);
            std::vector<size_t> new_col_idx;
            std::vector<double> new_values;

            for (size_t i = 0; i < rows_; ++i)
            {
                new_row_ptr[i + 1] = new_row_ptr[i];
                for (size_t j = row_ptr_[i]; j < row_ptr_[i + 1]; ++j)
                {
                    int col = col_idx_[j];
                    double val = values_[j];

                    bool found = false;
                    for (size_t k = new_row_ptr[i]; k < new_row_ptr[i + 1]; ++k)
                    {
                        if (new_col_idx[k] == col)
                        {
                            new_values[k] += val;
                            found = true;
                            break;
                        }
                    }

                    if (!found)
                    {
                        new_col_idx.push_back(col);
                        new_values.push_back(val);
                        ++new_row_ptr[i + 1];
                    }
                }
            }
            row_ptr_ = std::move(new_row_ptr);
            col_idx_ = std::move(new_col_idx);
            values_ = std::move(new_values);
        }

        double operator()(size_t i, size_t j) const override
        {
            if (i >= rows_ || j >= cols_)
                return 0.0;
            for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
            {
                if (col_idx_[k] == static_cast<int>(j))
                {
                    return values_[k];
                }
            }
            return 0.0;
        }

        void set(size_t i, size_t j, double val) override
        {
            if (i >= rows_ || j >= cols_)
                return;
            for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
            {
                if (col_idx_[k] == static_cast<int>(j))
                {
                    values_[k] = val;
                    return;
                }
            }

            throw std::runtime_error("set: entry not found");
        }

        void add(size_t i, size_t j, double val) override
        {
            if (i >= rows_ || j >= cols_)
                return;
            for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
            {
                if (col_idx_[k] == static_cast<int>(j))
                {
                    values_[k] += val;
                    return;
                }
            }
        }

        void clear() override
        {
            row_ptr_.assign(rows_ + 1, 0);
            col_idx_.clear();
            values_.clear();
        }

        void scale(double factor) override
        {
            for (auto &v : values_)
            {
                v *= factor;
            }
        }

        void mv(const std::vector<double> &x, std::vector<double> &y) const
        {
            y.assign(rows_, 0.0);
            for (size_t i = 0; i < rows_; ++i)
            {
                double sum = 0.0;
                for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
                {
                    sum += values_[k] * x[col_idx_[k]];
                }
                y[i] = sum;
            }
        }

        void mv(double alpha, const std::vector<double> &x,
                double beta, std::vector<double> &y) const
        {
            for (size_t i = 0; i < rows_; ++i)
            {
                double sum = 0.0;
                for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
                {
                    sum += values_[k] * x[col_idx_[k]];
                }
                y[i] = alpha * sum + beta * y[i];
            }
        }

        void mtv(const std::vector<double> &x, std::vector<double> &y) const
        {
            y.assign(cols_, 0.0);
            for (size_t i = 0; i < rows_; ++i)
            {
                for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
                {
                    y[col_idx_[k]] += values_[k] * x[i];
                }
            }
        }

        std::vector<std::tuple<int, double>> get_row(size_t row) const
        {
            std::vector<std::tuple<int, double>> result;
            for (size_t k = row_ptr_[row]; k < row_ptr_[row + 1]; ++k)
            {
                result.emplace_back(col_idx_[k], values_[k]);
            }
            return result;
        }

        std::vector<double> diagonal() const
        {
            std::vector<double> diag;
            diag.reserve(std::min(rows_, cols_));

            size_t min_dim = std::min(rows_, cols_);
            for (size_t i = 0; i < min_dim; ++i)
            {
                diag.push_back((*this)(i, i));
            }
            return diag;
        }

        double norm_frobenius() const
        {
            double sum = 0.0;
            for (const auto &v : values_)
            {
                sum += v * v;
            }
            return std::sqrt(sum);
        }

        void print() const
        {
            for (size_t i = 0; i < rows_; ++i)
            {
                for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
                {
                    std::cout << "(" << i << "," << col_idx_[k] << ")" << values_[k] << " ";
                }
                std::cout << "\n";
            }
        }

        const std::vector<size_t> &row_ptr() const { return row_ptr_; }
        const std::vector<size_t> &col_idx() const { return col_idx_; }
        const std::vector<double> &values() const { return values_; }

    private:
        size_t rows_;
        size_t cols_;
        std::vector<size_t> row_ptr_;
        std::vector<size_t> col_idx_;
        std::vector<double> values_;
    };

    class SparseMatrixCOO
    {
    public:
        explicit SparseMatrixCOO(size_t rows, size_t cols) : rows_(rows), cols_(cols), rows_idx_(), cols_idx_(), values_(), map_() {}
        ~SparseMatrixCOO() = default;
        void reserve(size_t nnz)
        {
            rows_idx_.reserve(nnz);
            cols_idx_.reserve(nnz);
            values_.reserve(nnz);
        }

        void add(size_t i, size_t j, double val)
        {
            auto key = std::make_pair(static_cast<int>(i), static_cast<int>(j));
            auto it = map_.find(key);
            if (it != map_.end())
            {
                it->second += val;
            }
            else
            {
                map_[key] = val;
                rows_idx_.push_back(static_cast<int>(i));
                cols_idx_.push_back(static_cast<int>(j));
                values_.push_back(val);
            }
        }

        void add_uique(size_t i, size_t j, double val)
        {
            auto it = map_.find(std::make_pair(static_cast<int>(i), static_cast<int>(j)));
            if (it != map_.end())
            {
                it->second += val;
            }
            else
            {
                map_[std::make_pair(static_cast<int>(i), static_cast<int>(j))] = val;
                rows_idx_.push_back(static_cast<int>(i));
                cols_idx_.push_back(static_cast<int>(j));
                values_.push_back(val);
            }
        }

        void clear()
        {
            rows_idx_.clear();
            cols_idx_.clear();
            values_.clear();
            map_.clear();
        }

        size_t rows() const { return rows_; }
        size_t cols() const { return cols_; }
        size_t nnz() const { return values_.size(); }

        SparseMatrixCRS to_crs() const
        {
            if (rows_idx_.empty())
            {
                return SparseMatrixCRS();
            }

            // non-zero in every row
            std::vector<size_t> row_count(rows_, 0);
            for (const auto &r : rows_idx_)
            {
                ++row_count[r];
            }
            // construct offset
            std::vector<size_t> row_ptr(rows_ + 1, 0);
            for (size_t i = 0; i < rows_; ++i)
            {
                row_ptr[i + 1] = row_ptr[i] + row_count[i];
            }

            std::vector<size_t> col_idx(values_.size());
            std::vector<double> vals(values_.size());
            std::vector<size_t> rows_ptr = row_ptr;

            for (size_t i = 0; i < rows_idx_.size(); ++i)
            {
                int row = rows_idx_[i];
                int idx = rows_ptr[row]++;
                col_idx[idx] = cols_idx_[i];
                vals[idx] = values_[i];
            }

            for (size_t i = 0; i < rows_; ++i)
            {
                std::sort(col_idx.begin() + row_ptr[i], col_idx.begin() + row_ptr[i + 1]);
                std::sort(vals.begin() + row_ptr[i], vals.begin() + row_ptr[i + 1]);
            }

            std::vector<size_t> new_row_ptr(rows_ + 1, 0);
            std::vector<size_t> new_col_idx;
            std::vector<double> new_vals;

            for (size_t i = 0; i < rows_; ++i)
            {
                new_col_idx.push_back(col_idx[row_ptr[i]]);
                new_vals.push_back(vals[row_ptr[i]]);

                for (size_t j = row_ptr[i] + 1; j < row_ptr[i + 1]; ++i)
                {
                    if (col_idx[j] == new_col_idx.back())
                    {
                        new_vals.back() += vals[j];
                    }
                    else
                    {
                        new_col_idx.push_back(col_idx[j]);
                        new_vals.push_back(vals[j]);
                    }
                }
                new_row_ptr[i + 1] = new_col_idx.size();
            }

            SparseMatrixCRS result;
            result.build(rows_, cols_, std::move(new_row_ptr), std::move(new_col_idx), std::move(new_vals));

            return result;
        }

    private:
        size_t rows_, cols_;
        std::vector<size_t> rows_idx_;
        std::vector<size_t> cols_idx_;
        std::vector<double> values_;
        std::unordered_map<std::pair<size_t, size_t>, double, PairHash<size_t>> map_;
    };

    template <typename T>
    struct PairHash
    {
        size_t operator()(const std::pair<T, T> &p) const noexcept
        {
            return std::hash<T>{}(p.first) ^ (std::hash<T>{}(p.second) << 1);
        }
    };
}