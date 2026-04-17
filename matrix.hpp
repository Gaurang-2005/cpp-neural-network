#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <thread>

template<typename T>
class matrix {
    size_t row;
    size_t col;
    T* mat;

public:
    size_t blockSize = 64;
    matrix() : row{0}, col{0}, mat{nullptr} {}

    matrix(size_t a, size_t b, double val = 0): row{a}, col{b} {
        mat = new T[row * col]{val};
    }

    //copy constructor
    matrix(const matrix<T>& temp) : row{temp.row}, col{temp.col} {
        mat = new T[row * col];
        for (size_t i = 0; i < row * col; i++) {
            mat[i] = temp.mat[i];
        }
    }

    //move constructor
    matrix(matrix<T>&& temp) noexcept : row{temp.row}, col{temp.col} {
        temp.row = 0;
        temp.col = 0;

        mat = temp.mat;
        temp.mat = nullptr;
    }

    //move assignment
    matrix<T>& operator=(matrix<T>&& temp) noexcept {
        if (&temp == this) return *this;
        delete[] mat;
        mat = temp.mat;
        row = temp.row;
        col = temp.col;
        temp.mat = nullptr;
        temp.row = 0;
        temp.col = 0;

        return *this;
    }

    //copy assignment
    matrix<T>& operator=(const matrix<T>& temp) {
        if (&temp == this) return *this;
        T* temp2 = new T[temp.row * temp.col];
        for (size_t i = 0; i < temp.row * temp.col; i++) {
            temp2[i] = temp.mat[i];
        }
        delete[] mat;
        mat = temp2;
        row = temp.row;
        col = temp.col;

        return *this;
    }

    //number of rows getter
    size_t rows() const {
        return row;
    }

    //number of columns getter
    size_t cols() const {
        return col;
    }

    T& operator()(size_t a, size_t b) {
        assert(a < row && b < col);
        return mat[a * (col) + b];
    }
    matrix<T> operator()(size_t a) const {
        assert(a < col);
        matrix<T> temp(1, col);
        for (size_t i = 0; i < col; i++) {
            temp.mat[i] = mat[a * col + i];
        }
        return temp;
    }
    const T& operator()(size_t a, size_t b) const {
        assert(a < row && b < col);
        return mat[a * (col) + b];
    }

    matrix<T> operator+(const matrix<T>& second) const {
        assert(col == second.col && row == second.row);
        matrix<T> temp(row, col);

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                temp.mat[i * col + j] = mat[i * col + j] + second.mat[i * col + j];
            }
        }

        return temp;
    } 

    matrix<T>& operator+=(const matrix<T>& second) {
        assert(col == second.col && row == second.row);

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                mat[i * col + j] += second.mat[i * col + j];
            }
        }

        return *this;
    } 
    matrix<T> operator-(const matrix<T>& second) const {
        assert(col == second.col && row == second.row);
        matrix<T> temp(row, col);

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                temp.mat[i * col + j] = mat[i * col + j] - second.mat[i * col + j];
            }
        }

        return temp;
    } 

    matrix<T>& operator-=(const matrix<T>& second) {
        assert(col == second.col && row == second.row);

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                mat[i * col + j] -= second.mat[i * col + j];
            }
        }

        return *this;
    }     
    matrix<T> hadamardProduct(const matrix<T>& second) const {
        assert(col == second.col && row == second.row);
        matrix<T> temp(row, col);
        for (size_t i = 0; i < row * col; i++) temp.mat[i] = mat[i] * second.mat[i];
        
        return temp;
    }
    void MultiplicationLoop(matrix<T>& result, const matrix<T>& second, size_t blockSize, size_t blockX1, size_t blockY1, size_t blockX2, size_t blockY2) const {
        
        size_t maxX = std::min(second.row, blockSize * (blockX2 + 1)) - blockSize * blockX2;
        size_t maxY = std::min(row, blockSize * (blockY1 + 1)) - blockSize * blockY1;
        size_t common = std::min(col, blockSize * (blockX1 + 1)) - blockSize * blockX1;

        size_t ir1 = blockY1 * blockSize;
        size_t jr1 = blockX1 * blockSize;
        size_t ir2 = blockY2 * blockSize;
        size_t jr2 = blockX2 * blockSize;        
        
        for (size_t i = 0; i < maxY; i++) {
            for (size_t k = 0; k < common; k++) {
                T temp = mat[(ir1 + i) * col + jr1 + k];
                for (size_t j = 0; j < maxX; j++) {
                    result.mat[(ir1 + i) * result.col + jr2 + j] += temp * second.mat[(jr2 + j) * second.col + ir2 + k];
                }
            }
        }
    }

    void blockRange(matrix<T>& result, const matrix<T>& second, size_t blockSize, size_t blockCom, size_t blockRow, size_t iniR, size_t finR) const {

        for (size_t i = iniR; i < finR; i++) {
            for (size_t k = 0; k < blockCom; k++) {
                MultiplicationLoop(result, second, blockSize, k, i / blockRow, i % blockRow, k);
            }
        }
    }

    //With Blocking:
    matrix<T> operator*(const matrix<T>& second) const {
        assert(col == second.row);
        int hardThreads = std::thread::hardware_concurrency();
        hardThreads = (hardThreads > 0) ? hardThreads : 1;

        matrix<T> secondT = second.transpose();

        size_t blockX1 = (col % blockSize == 0)? col / blockSize : col / blockSize + 1;
        size_t blockY1 = (row % blockSize == 0)? row / blockSize : row / blockSize + 1;
        size_t blockX2 = (second.col % blockSize == 0)? second.col / blockSize : second.col / blockSize + 1;
        // size_t blockY2 = (second.row % blockSize == 0)? second.row / blockSize : second.row / blockSize + 1;
        size_t blockRX = blockX2;
        size_t blockRY = blockY1;

        unsigned int Rblocks = (blockRX * blockRY);
        unsigned int maxThreads = hardThreads;
        if (second.col * row <= 10000) maxThreads = 1;
        
        matrix<T> result(row, second.col);
        std::unique_ptr<std::thread[]> threadArr (new std::thread[maxThreads]);
        unsigned int blocksPerThread = Rblocks / maxThreads;
        for (unsigned int i = 0; i < maxThreads; i++) {
            threadArr[i] = std::thread(&matrix<T>::blockRange, this, std::ref(result), std::ref(secondT), blockSize, blockX1, blockRX, i * blocksPerThread, (i == maxThreads - 1)?(i + 1) * blocksPerThread + Rblocks % maxThreads:(i + 1) * blocksPerThread); 
        }

        for (unsigned int i = 0; i < maxThreads; i++) threadArr[i].join();

        return result;
    }
    
    matrix<T>& operator*=(const matrix<T>& second) {

        *this = *this * second;

        return *this;
    }
    matrix<T> operator*(T num) const {
        matrix<T> result(row, col);

        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                result.mat[i * col + j] = mat[i * col + j] * num;
            }
        }

        return result;
    }

    matrix<T>& operator*=(T num) {

        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                mat[i * col + j] *= num;
            }
        }

        return *this;
    }
    
    friend matrix<T> operator*(const T num, const matrix<T>& temp) {
        matrix<T> result(temp.row, temp.col);

        for (size_t i = 0; i < temp.row; i++) {
            for (size_t j = 0; j < temp.col; j++) {
                result.mat[i * result.col + j] = temp.mat[i * temp.col + j] * num;
            }
        }

        return result;
    }

    matrix<T> transpose() const {
        matrix<T> result(col, row);

        for (size_t i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                result.mat[j * result.col + i] = mat[i * col + j];
            }
        }

        return result;
    }

    ~matrix() noexcept {
        delete[] mat;
    }
};