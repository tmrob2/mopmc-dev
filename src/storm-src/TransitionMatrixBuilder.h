//
// Created by thomas on 3/08/23.
//

#ifndef MOPMC_DEV_TRANSITIONMATRIXBUILDER_H
#define MOPMC_DEV_TRANSITIONMATRIXBUILDER_H


template <typename IndexType, typename ValueType>
class MatrixEntry {
public:
    typedef IndexType index_type;
    typedef ValueType value_type;

    MatrixEntry(index_type column, value_type value);

    MatrixEntry(std::pair<index_type, value_type>&& pair);

    MatrixEntry() = default;
    MatrixEntry(MatrixEntry const& other) = default;
    MatrixEntry& operator=(MatrixEntry const& other) = default;

    index_type const& getColumn() const;

    void setColumn(index_type const& column);

    value_type const& getValue() const;

    void setValue(value_type const& value);

    std::pair<index_type, value_type> const& getColumnValuePair() const;

    MatrixEntry operator*(value_type factor) const;

    bool operator==(MatrixEntry const& other) const;
    bool operator!=(MatrixEntry const& other) const;

    template<typename IndexTypePrime, typename ValueTypePrime>
    friend std::ostream& operator<<(std::ostream& out, MatrixEntry<IndexTypePrime, ValueTypePrime> const& entry);

private:
    // The actual matrix entry.
    std::pair<index_type, value_type> entry;
};
#endif //MOPMC_DEV_TRANSITIONMATRIXBUILDER_H
