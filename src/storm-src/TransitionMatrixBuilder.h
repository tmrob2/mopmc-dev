//
// Created by thomas on 3/08/23.
//

#ifndef MOPMC_DEV_TRANSITIONMATRIXBUILDER_H
#define MOPMC_DEV_TRANSITIONMATRIXBUILDER_H

#include "storm/adapters/RationalFunctionAdapter.h"
#include <storm/storage/sparse/StateType.h>

#include <storm/solver/OptimizationDirection.h>
#include <storm/storage/BitVector.h>
#include <storm/storage/sparse/StateType.h>

#include <storm/adapters/IntelTbbAdapter.h>
#include <storm/utility/OsDetection.h>
#include <storm/utility/constants.h>

#include <vector>
#include <algorithm>
#include <cstdint>
#include <iosfwd>
#include <iterator>
#include <boost/optional.hpp>

typedef storm::storage::sparse::state_type SparseMatrixIndexType;

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

class SparseMatrixBuilder {
public:
    typedef SparseMatrixIndexType index_type;

    /*!
     * Constructs a sparse matrix builder producing a matrix with the given number of rows, columns and entries.
     * The number of rows, columns and entries is reserved upon creation. If more rows/columns or entries are
     * added, this will possibly lead to a reallocation.
     *
     * @param rows The number of rows of the resulting matrix.
     * @param columns The number of columns of the resulting matrix.
     * @param entries The number of entries of the resulting matrix.
     * @param forceDimensions If this flag is set, the matrix is expected to have exactly the given number of
     * rows, columns and entries for all of these entities that are set to a nonzero value.
     * @param hasCustomRowGrouping A flag indicating whether the builder is used to create a non-canonical
     * grouping of rows for this matrix.
     * @param rowGroups The number of row groups of the resulting matrix. This is only relevant if the matrix
     * has a custom row grouping.
     */
    SparseMatrixBuilder(index_type rows, index_type columns, index_type entries, bool forceDimensions, bool hasCustomRowGrouping,
                                                    index_type rowGroups) 
    : 
        initialRowCountSet(rows != 0),
        initialRowCount(rows),
        initialColumnCountSet(columns != 0),
        initialColumnCount(columns),
        initialEntryCountSet(entries != 0),
        initialEntryCount(entries),
        forceInitialDimensions(forceDimensions),
        hasCustomRowGrouping(hasCustomRowGrouping),
        initialRowGroupCountSet(rowGroups != 0),
        initialRowGroupCount(rowGroups),
        rowGroupIndices(),
        columnsAndValues(),
        rowIndications(),
        currentEntryCount(0),
        lastRow(0),
        lastColumn(0),
        highestColumn(0),
        currentRowGroupCount(0) 
    {
        // Prepare the internal storage.
        if (initialRowCountSet) {
            rowIndications.reserve(initialRowCount + 1);
        }
        if (initialEntryCountSet) {
            columnsAndValues.reserve(initialEntryCount);
        }
        if (hasCustomRowGrouping) {
            rowGroupIndices = std::vector<index_type>();
        }
        if (initialRowGroupCountSet && hasCustomRowGrouping) {
            rowGroupIndices.get().reserve(initialRowGroupCount + 1);
        }
        rowIndications.push_back(0);
    }

    void addNextValue(index_type row, index_type col, double const& val);

private:
    // A flag indicating whether a row count was set upon construction.
    bool initialRowCountSet;

    // The row count that was initially set (if any).
    index_type initialRowCount;

    // A flag indicating whether a column count was set upon construction.
    bool initialColumnCountSet;

    // The column count that was initially set (if any).
    index_type initialColumnCount;

    // A flag indicating whether an entry count was set upon construction.
    bool initialEntryCountSet;

    // The number of entries in the matrix.
    index_type initialEntryCount;

    // A flag indicating whether the initially given dimensions are to be enforced on the resulting matrix.
    bool forceInitialDimensions;

    // A flag indicating whether the builder is to construct a custom row grouping for the matrix.
    bool hasCustomRowGrouping;

    // A flag indicating whether the number of row groups was set upon construction.
    bool initialRowGroupCountSet;

    // The number of row groups in the matrix.
    index_type initialRowGroupCount;

    // The vector that stores the row-group indices (if they are non-trivial).
    boost::optional<std::vector<index_type>> rowGroupIndices;

    // The storage for the columns and values of all entries in the matrix.
    std::vector<MatrixEntry<index_type, double>> columnsAndValues;

    // A vector containing the indices at which each given row begins. This index is to be interpreted as an
    // index in the valueStorage and the columnIndications vectors. Put differently, the values of the entries
    // in row i are valueStorage[rowIndications[i]] to valueStorage[rowIndications[i + 1]] where the last
    // entry is not included anymore.
    std::vector<index_type> rowIndications;

    // Stores the current number of entries in the matrix. This is used for inserting an entry into a matrix
    // with preallocated storage.
    index_type currentEntryCount;

    // Stores the row of the last entry in the matrix. This is used for correctly inserting an entry into a
    // matrix.
    index_type lastRow;

    // Stores the column of the currently last entry in the matrix. This is used for correctly inserting an
    // entry into a matrix.
    index_type lastColumn;

    // Stores the highest column at which an entry was inserted into the matrix.
    index_type highestColumn;

    // Stores the currently active row group. This is used for correctly constructing the row grouping of the
    // matrix.
    index_type currentRowGroupCount;

    boost::optional<double> pendingDiagonalEntry;
};
#endif //MOPMC_DEV_TRANSITIONMATRIXBUILDER_H
