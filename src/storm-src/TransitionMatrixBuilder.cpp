//
// Created by thomas on 2/08/23.
//
#include "TransitionMatrixBuilder.h"

template<typename IndexType, typename ValueType>
IndexType const& MatrixEntry<IndexType, ValueType>::getColumn() const {
    return this -> entry.first;
}

template<typename IndexType, typename ValueType>
void MatrixEntry<IndexType, ValueType>::setColumn(const IndexType& column) {
    this -> entry.first = column;
}

template<typename IndexType, typename ValueType>
const ValueType& MatrixEntry<IndexType, ValueType>::getValue() const {
    return this -> entry.second;
}

template<typename IndexType, typename ValueType>
void MatrixEntry<IndexType, ValueType>::setValue(const ValueType &value) {
    this -> entry.second = value;
}

template<typename IndexType, typename ValueType>
std::pair<IndexType, ValueType> const& MatrixEntry<IndexType, ValueType>::getColumnValuePair() const {
    return this->entry;
}

template<typename IndexType, typename ValueType>
MatrixEntry<IndexType, ValueType> MatrixEntry<IndexType, ValueType>::operator*(MatrixEntry::value_type factor) const {
    return MatrixEntry(this->getColumn(), this->getValue() * factor);
}

template<typename IndexType, typename ValueType>
bool MatrixEntry<IndexType, ValueType>::operator==(
        const MatrixEntry<IndexType, ValueType> &other) const {
    return this->entry.first == other.entry.first && this-> entry.second == other.entry.second;
}

template<typename IndexType, typename ValueType>
bool MatrixEntry<IndexType, ValueType>::operator!=(const MatrixEntry<IndexType, ValueType> &other) const {
    return *this != other;
}

template<typename IndexTypePrime, typename ValueTypePrime>
std::ostream& operator<<(std::ostream& out, MatrixEntry<IndexTypePrime, ValueTypePrime> const&entry) {
    out << "(" << entry.getColumn() << ", " << entry.getValue() << ")";
    return out;
}












