/********************************************************************************
 * Copyright (c) 2025 Contributors to the Eclipse Foundation
 *
 * See the NOTICE file(s) distributed with this work for additional
 * information regarding copyright ownership.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * SPDX-License-Identifier: Apache-2.0
 ********************************************************************************/
#include "score/concurrency/future/interruptible_future.h"
#include "score/concurrency/future/interruptible_promise.h"

#include "score/stop_token.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <future>
#include <type_traits>
#include <utility>

namespace score
{
namespace concurrency
{
namespace detail
{
namespace
{

TEST(InterruptibleStateTest, Destruction)
{
    /*
     * BaseInterruptibleState inherits from std::enable_shared_from_this. Thus, it should never (!) be allocated on the
     * stack or via a unique_ptr on the heap!
     *
     * Because GCC emits symbols for all constructors and destructors defined in the Itanium C++ ABI independent of
     * their usage, we need to test these to achieve the required function coverage (100%). Hence, for this test, we
     * ignore above requirement and construct on stack, and on heap with unique pointers.
     *
     * Specifically, we use unique pointers because constructing a shared pointer using std::make_shared will not (!)
     * call the deleting destructor (D0) of BaseInterruptibleState. Instead, it will call the deleting destructor of
     * std::shared_ptr and that will in turn call the complete object destructor (D1) of BaseInterruptibleState.
     */

    // Required to cover D0 destructor
    auto heap_base_state = std::make_unique<InterruptibleState<void>>();
    heap_base_state.reset();

    // Required to cover D1 destructors
    {
        InterruptibleState<void> stack_base_state{};
        (void)stack_base_state;
    }

    // Required to cover D2 destructor
    std::shared_ptr<InterruptibleState<void>> shared_heap_base_state;
    {
        shared_heap_base_state = std::make_shared<InterruptibleState<void>>();
    }
    shared_heap_base_state.reset();
}

// Test SetValue with move-constructible type (int)
TEST(InterruptibleStateSetValueTest, SetValueMoveInt)
{
    auto state = InterruptibleState<int>::Make();
    int value = 42;
    EXPECT_TRUE(state->SetValue(std::move(value)));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value(), 42);
}

// Test SetValue with copy-constructible type (int)
TEST(InterruptibleStateSetValueTest, SetValueCopyInt)
{
    auto state = InterruptibleState<int>::Make();
    const int value = 42;
    EXPECT_TRUE(state->SetValue(value));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value(), 42);
}

// Note: Idempotent behavior (SetValue returns false on second call) is tested comprehensively
// in SetValueReturnValueConsistency and SetValueMultipleRapidAttempts, so type-specific
// "TwiceReturnsFalse" tests are consolidated into those generic tests.

// Test SetValue with move-only type
TEST(InterruptibleStateSetValueTest, SetValueMoveOnly)
{
    auto state = InterruptibleState<std::unique_ptr<int>>::Make();
    auto value = std::make_unique<int>(42);
    EXPECT_TRUE(state->SetValue(std::move(value)));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(*state->GetValue().value(), 42);
}

// Test SetValue with string
TEST(InterruptibleStateSetValueTest, SetValueString)
{
    auto state = InterruptibleState<std::string>::Make();
    std::string value = "hello";
    EXPECT_TRUE(state->SetValue(std::move(value)));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value(), "hello");
}

// Test SetValue with reference type (lvalue reference)
TEST(InterruptibleStateSetValueTest, SetValueReference)
{
    auto state = InterruptibleState<int&>::Make();
    int value = 42;
    EXPECT_TRUE(state->SetValue(value));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().get(), 42);
}

// Test SetValue with void specialization
TEST(InterruptibleStateSetValueTest, SetValueVoid)
{
    auto state = InterruptibleState<void>::Make();
    EXPECT_TRUE(state->SetValue());
    EXPECT_TRUE(state->GetValue().has_value());
}

// Test SetValue with custom type
struct CustomType
{
    int x;
    std::string y;

    bool operator==(const CustomType& other) const
    {
        return x == other.x && y == other.y;
    }
};

TEST(InterruptibleStateSetValueTest, SetValueCustomType)
{
    auto state = InterruptibleState<CustomType>::Make();
    CustomType value{42, "test"};
    EXPECT_TRUE(state->SetValue(std::move(value)));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().x, 42);
    EXPECT_EQ(state->GetValue().value().y, "test");
}

// Note: SetError/SetValue interaction comprehensively tested in SetValueSetErrorOrder

class MoveOnly
{
  public:
    explicit MoveOnly(int id) noexcept : id_(id) {}

    MoveOnly(const MoveOnly&) = delete;
    MoveOnly& operator=(const MoveOnly&) = delete;

    MoveOnly(MoveOnly&& other) noexcept : id_(other.id_)
    {
        other.id_ = -1;
    }

    MoveOnly& operator=(MoveOnly&& other) noexcept
    {
        id_ = other.id_;
        other.id_ = -1;
        return *this;
    }

    int GetId() const noexcept
    {
        return id_;
    }

  private:
    int id_;
};

template <typename T>
class Counted
{
  public:
    template <typename... Args,
              std::enable_if_t<std::is_constructible<T, Args&&...>::value, bool> = true>
    explicit Counted(Args&&... args) : value_(std::forward<Args>(args)...)
    {
        ++construction_count_;
    }

    Counted(const Counted&) = delete;
    Counted& operator=(const Counted&) = delete;

    Counted(Counted&& other) noexcept(std::is_nothrow_move_constructible<T>::value)
        : value_(std::move(other.value_))
    {
        ++construction_count_;
    }

    Counted& operator=(Counted&&) = delete;

    ~Counted()
    {
        ++destruction_count_;
    }

    T& GetValue() noexcept
    {
        return value_;
    }

    const T& GetValue() const noexcept
    {
        return value_;
    }

    static void ResetCounts() noexcept
    {
        construction_count_ = 0;
        destruction_count_ = 0;
    }

    static int ConstructionCount() noexcept
    {
        return construction_count_;
    }

    static int DestructionCount() noexcept
    {
        return destruction_count_;
    }

  private:
    T value_;
    inline static int construction_count_{0};
    inline static int destruction_count_{0};
};

TEST(InterruptibleStateSetValueTest, SetValueCountedMoveOnlyFromInt)
{
    using Value = Counted<MoveOnly>;
    using V = int;

    Value::ResetCounts();

    auto state = InterruptibleState<Value>::Make();
    EXPECT_TRUE(state->SetValue(V{42}));
    ASSERT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().GetValue().GetId(), 42);
    EXPECT_EQ(Value::ConstructionCount(), 1);
    EXPECT_EQ(Value::DestructionCount(), 0);
}

TEST(InterruptibleStateSetValueTest, SetValueCountedMoveOnlyDestroysStoredValue)
{
    using Value = Counted<MoveOnly>;

    Value::ResetCounts();
    {
        auto state = InterruptibleState<Value>::Make();
        Value value{42};

        EXPECT_TRUE(state->SetValue(std::move(value)));
        EXPECT_TRUE(state->GetValue().has_value());
        EXPECT_EQ(state->GetValue().value().GetValue().GetId(), 42);
        EXPECT_EQ(Value::ConstructionCount(), 2);
        EXPECT_EQ(Value::DestructionCount(), 0);
    }

    EXPECT_EQ(Value::DestructionCount(), 2);
}

TEST(InterruptibleStateSetValueTest, SetValueCountedMoveOnlyPreservesStoredValue)
{
    using Value = Counted<MoveOnly>;

    Value::ResetCounts();
    auto state = InterruptibleState<Value>::Make();
    Value value{42};

    EXPECT_TRUE(state->SetValue(std::move(value)));
    EXPECT_FALSE(state->SetValue(Value{100}));
    EXPECT_EQ(state->GetValue().value().GetValue().GetId(), 42);
}

// Non-assignable type: has constructors but deleted assignment operators
// This demonstrates the need for placement new in SetValue
class NonAssignable
{
  public:
    explicit NonAssignable(int value) : value_(value) {}

    // Allow moves but not assignment
    NonAssignable(NonAssignable&& other) noexcept : value_(other.value_) {}

    // Allow copies but not assignment
    NonAssignable(const NonAssignable& other) : value_(other.value_) {}

    // Delete all assignment operators
    NonAssignable& operator=(const NonAssignable&) = delete;
    NonAssignable& operator=(NonAssignable&&) = delete;

    int GetValue() const { return value_; }

  private:
    int value_;
};

// Test SetValue with non-assignable move-constructible type
TEST(InterruptibleStateSetValueTest, SetValueNonAssignableMove)
{
    auto state = InterruptibleState<NonAssignable>::Make();
    NonAssignable value{42};

    // This works because SetValue uses placement new, not assignment
    EXPECT_TRUE(state->SetValue(std::move(value)));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().GetValue(), 42);
}

// Test SetValue with non-assignable copy-constructible type
TEST(InterruptibleStateSetValueTest, SetValueNonAssignableCopy)
{
    auto state = InterruptibleState<NonAssignable>::Make();
    const NonAssignable value{100};

    // This works because SetValue uses placement new, not assignment
    EXPECT_TRUE(state->SetValue(value));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().GetValue(), 100);
}

// Test SetValue with non-assignable type returns false on duplicate
TEST(InterruptibleStateSetValueTest, SetValueNonAssignableTwiceReturnsFalse)
{
    auto state = InterruptibleState<NonAssignable>::Make();
    NonAssignable value1{42};

    EXPECT_TRUE(state->SetValue(std::move(value1)));
    EXPECT_EQ(state->GetValue().value().GetValue(), 42);

    // Second SetValue fails
    NonAssignable value2{100};
    EXPECT_FALSE(state->SetValue(std::move(value2)));

    // Original value unchanged
    EXPECT_EQ(state->GetValue().value().GetValue(), 42);
}

// Test SetValue with reference to non-assignable type
TEST(InterruptibleStateSetValueTest, SetValueNonAssignableReference)
{
    auto state = InterruptibleState<NonAssignable&>::Make();
    NonAssignable value{75};

    // Reference specialization stores reference_wrapper
    EXPECT_TRUE(state->SetValue(value));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().get().GetValue(), 75);
}

// Test SetValue preserves value across GetValue calls
TEST(InterruptibleStateSetValueTest, SetValuePreservesValueConsistency)
{
    auto state = InterruptibleState<int>::Make();
    EXPECT_TRUE(state->SetValue(42));

    // Multiple GetValue calls should return the same value
    EXPECT_EQ(state->GetValue().value(), 42);
    EXPECT_EQ(state->GetValue().value(), 42);
    EXPECT_EQ(state->GetValue().value(), 42);

    // Verify state_ reference is stable
    EXPECT_EQ(&state->GetValue(), &state->GetValue());
}

// Test SetValue with large string type
TEST(InterruptibleStateSetValueTest, SetValueLargeString)
{
    auto state = InterruptibleState<std::string>::Make();
    std::string large_value(10000, 'x');  // 10KB string

    EXPECT_TRUE(state->SetValue(std::move(large_value)));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().size(), 10000);
    EXPECT_EQ(state->GetValue().value().back(), 'x');
}

// Test SetValue with nested container type
TEST(InterruptibleStateSetValueTest, SetValueNestedContainer)
{
    auto state = InterruptibleState<std::vector<std::vector<int>>>::Make();
    std::vector<std::vector<int>> nested{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    EXPECT_TRUE(state->SetValue(std::move(nested)));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().size(), 3);
    EXPECT_EQ(state->GetValue().value()[0][0], 1);
    EXPECT_EQ(state->GetValue().value()[2][2], 9);
}

// Test SetValue multiple times in rapid succession
TEST(InterruptibleStateSetValueTest, SetValueMultipleRapidAttempts)
{
    auto state = InterruptibleState<int>::Make();

    // First succeeds
    EXPECT_TRUE(state->SetValue(10));

    // Rapid successive attempts all fail
    for (int i = 0; i < 100; ++i)
    {
        EXPECT_FALSE(state->SetValue(i));
    }

    // Original value still there
    EXPECT_EQ(state->GetValue().value(), 10);
}

// Test SetValue interleaved with GetValue
TEST(InterruptibleStateSetValueTest, SetValueInterleamedWithGetValue)
{
    auto state = InterruptibleState<int>::Make();

    EXPECT_TRUE(state->SetValue(42));
    EXPECT_EQ(state->GetValue().value(), 42);

    // Try to set again
    EXPECT_FALSE(state->SetValue(100));
    EXPECT_EQ(state->GetValue().value(), 42);

    // Value unchanged
    EXPECT_EQ(state->GetValue().value(), 42);
}

// Test SetValue with pair type
TEST(InterruptibleStateSetValueTest, SetValuePairType)
{
    auto state = InterruptibleState<std::pair<int, std::string>>::Make();
    std::pair<int, std::string> value{42, "hello"};

    EXPECT_TRUE(state->SetValue(std::move(value)));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().first, 42);
    EXPECT_EQ(state->GetValue().value().second, "hello");
}

// Test SetValue return value is consistent
TEST(InterruptibleStateSetValueTest, SetValueReturnValueConsistency)
{
    auto state = InterruptibleState<int>::Make();

    // First call returns true
    bool result1 = state->SetValue(1);
    EXPECT_TRUE(result1);

    // All subsequent calls return false
    bool result2 = state->SetValue(2);
    bool result3 = state->SetValue(3);
    EXPECT_FALSE(result2);
    EXPECT_FALSE(result3);
}

// Test SetValue doesn't invoke assignment operator
TEST(InterruptibleStateSetValueTest, SetValueAvoidsAssignmentOperator)
{
    // NonAssignable has constructors but deleted operator=
    auto state = InterruptibleState<NonAssignable>::Make();

    NonAssignable value1{111};
    EXPECT_TRUE(state->SetValue(std::move(value1)));

    // Second attempt fails without invoking assignment
    NonAssignable value2{222};
    EXPECT_FALSE(state->SetValue(std::move(value2)));

    // Original value intact
    EXPECT_EQ(state->GetValue().value().GetValue(), 111);
}

// Test SetValue with vector of move-only elements
TEST(InterruptibleStateSetValueTest, SetValueVectorOfMoveOnly)
{
    auto state = InterruptibleState<std::vector<std::unique_ptr<int>>>::Make();
    std::vector<std::unique_ptr<int>> values;
    values.push_back(std::make_unique<int>(1));
    values.push_back(std::make_unique<int>(2));
    values.push_back(std::make_unique<int>(3));

    EXPECT_TRUE(state->SetValue(std::move(values)));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().size(), 3);
    EXPECT_EQ(*state->GetValue().value()[0], 1);
    EXPECT_EQ(*state->GetValue().value()[2], 3);
}

// Test SetValue is not affected by SetError call order
TEST(InterruptibleStateSetValueTest, SetValueSetErrorOrder)
{
    auto state1 = InterruptibleState<int>::Make();
    auto state2 = InterruptibleState<int>::Make();

    // state1: SetValue first, then try SetError
    EXPECT_TRUE(state1->SetValue(100));
    EXPECT_FALSE(state1->SetError(Error::kUnset));
    EXPECT_EQ(state1->GetValue().value(), 100);

    // state2: SetError first, then try SetValue
    EXPECT_TRUE(state2->SetError(Error::kUnset));
    EXPECT_FALSE(state2->SetValue(200));
    EXPECT_FALSE(state2->GetValue().has_value());
}

// Test SetValue GetValue reference is the same
TEST(InterruptibleStateSetValueTest, SetValueGetValueReferenceStability)
{
    auto state = InterruptibleState<int>::Make();
    state->SetValue(42);

    score::Result<int>& ref1 = state->GetValue();
    score::Result<int>& ref2 = state->GetValue();

    // Same reference
    EXPECT_EQ(&ref1, &ref2);

    // Value accessible through both
    EXPECT_EQ(ref1.value(), 42);
    EXPECT_EQ(ref2.value(), 42);
}

// Test SetValue with empty container types
TEST(InterruptibleStateSetValueTest, SetValueEmptyContainers)
{
    auto state_vec = InterruptibleState<std::vector<int>>::Make();
    std::vector<int> empty_vec;
    EXPECT_TRUE(state_vec->SetValue(std::move(empty_vec)));
    EXPECT_TRUE(state_vec->GetValue().value().empty());

    auto state_str = InterruptibleState<std::string>::Make();
    std::string empty_str;
    EXPECT_TRUE(state_str->SetValue(std::move(empty_str)));
    EXPECT_TRUE(state_str->GetValue().value().empty());
}

// Test SetValue with const and non-const reference overload selection
TEST(InterruptibleStateSetValueTest, SetValueConstNonConstOverloadSelection)
{
    auto state1 = InterruptibleState<int>::Make();
    const int const_val = 42;
    EXPECT_TRUE(state1->SetValue(const_val));
    EXPECT_EQ(state1->GetValue().value(), 42);

    auto state2 = InterruptibleState<int>::Make();
    int non_const_val = 99;
    EXPECT_TRUE(state2->SetValue(std::move(non_const_val)));
    EXPECT_EQ(state2->GetValue().value(), 99);
}

// Test SetValue behavior with reference type modification
TEST(InterruptibleStateSetValueTest, SetValueReferenceTypeModification)
{
    auto state = InterruptibleState<int&>::Make();
    int original = 42;

    EXPECT_TRUE(state->SetValue(original));
    EXPECT_EQ(state->GetValue().value().get(), 42);

    // Modify original
    original = 100;

    // Reference now points to modified value
    EXPECT_EQ(state->GetValue().value().get(), 100);
}

// Base class for derived type testing
class BaseType
{
  public:
    explicit BaseType(int val) : value(val) {}
    int value;
};

// Derived class convertible to base
class DerivedType : public BaseType
{
  public:
    explicit DerivedType(int val) : BaseType(val) {}
};

// Test SetValue with derived type implicitly convertible to base (move)
TEST(InterruptibleStateSetValueTest, SetValueDerivedMoveToBase)
{
    auto state = InterruptibleState<BaseType>::Make();
    DerivedType derived{42};

    // Implicit conversion from DerivedType to BaseType via move constructor
    EXPECT_TRUE(state->SetValue(std::move(derived)));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().value, 42);
}

// Test SetValue with derived type implicitly convertible to base (copy)
TEST(InterruptibleStateSetValueTest, SetValueDerivedCopyToBase)
{
    auto state = InterruptibleState<BaseType>::Make();
    const DerivedType derived{99};

    // Implicit conversion from DerivedType to BaseType via copy constructor
    EXPECT_TRUE(state->SetValue(derived));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().value, 99);
}

// Test SetValue with smaller int type to larger (move)
TEST(InterruptibleStateSetValueTest, SetValueImplicitIntConversionMove)
{
    auto state = InterruptibleState<long>::Make();
    short short_val = 100;

    // Implicit conversion from short to long via move
    EXPECT_TRUE(state->SetValue(std::move(short_val)));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value(), 100L);
}

// Test SetValue with smaller int type to larger (copy)
TEST(InterruptibleStateSetValueTest, SetValueImplicitIntConversionCopy)
{
    auto state = InterruptibleState<long>::Make();
    const short short_val = 200;

    // Implicit conversion from short to long via copy
    EXPECT_TRUE(state->SetValue(short_val));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value(), 200L);
}

// Type with explicit converting constructor
class ConvertibleType
{
  public:
    explicit ConvertibleType(int val) : value(val) {}
    int value;
};

class ConvertsFromInt
{
  public:
    // Constructor that accepts ConvertibleType
    ConvertsFromInt(const ConvertibleType& other) : value(other.value) {}
    int value;
};

class MoveSource
{
    public:
        explicit MoveSource(int value) : value_(value) {}

        MoveSource(const MoveSource&) = delete;
        MoveSource& operator=(const MoveSource&) = delete;

        MoveSource(MoveSource&& other) noexcept : value_(other.value_)
        {
                other.value_ = 0;
        }

        MoveSource& operator=(MoveSource&&) = delete;

        int GetValue() const
        {
                return value_;
        }

    private:
        int value_;
};

class ValueConstructibleFromMoveSource
{
    public:
        explicit ValueConstructibleFromMoveSource(MoveSource&& source) noexcept : value_(source.GetValue()) {}

        ValueConstructibleFromMoveSource(const MoveSource&) = delete;

        int GetValue() const
        {
                return value_;
        }

    private:
        int value_;
};

class ThrowingValueConstructibleFromMoveSource
{
    public:
        explicit ThrowingValueConstructibleFromMoveSource(MoveSource&& source) noexcept(false)
                : value_(source.GetValue())
        {
        }

        ThrowingValueConstructibleFromMoveSource(const MoveSource&) = delete;

        int GetValue() const
        {
                return value_;
        }

    private:
        int value_;
};

// Test SetValue with type that has converting constructor (copy)
TEST(InterruptibleStateSetValueTest, SetValueConvertingConstructorCopy)
{
    auto state = InterruptibleState<ConvertsFromInt>::Make();
    ConvertibleType convertible{42};

    // V = ConvertibleType, Value = ConvertsFromInt
    // Uses converting constructor ConvertsFromInt(const ConvertibleType&)
    EXPECT_TRUE(state->SetValue(convertible));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().value, 42);
}

TEST(InterruptibleStateSetValueTest, SetValueAcceptsDistinctMoveArgumentType)
{
    using Value = ValueConstructibleFromMoveSource;
    using V = MoveSource;

    static_assert(!std::is_same<Value, V>::value, "The move argument type must differ from Value");
    static_assert(std::is_constructible<Value, V&&>::value, "Value must be constructible from V&&");
    static_assert(!std::is_constructible<Value, const V&>::value, "Value must not be constructible from const V&");

    auto state = InterruptibleState<Value>::Make();
    V source{42};

    EXPECT_TRUE(state->SetValue(std::move(source)));
    ASSERT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().GetValue(), 42);
}

TEST(InterruptibleStateSetValueTest, SetValueAcceptsNothrowConstructibleValue)
{
    using Value = ValueConstructibleFromMoveSource;
    using V = MoveSource;

    static_assert(std::is_nothrow_constructible<Value, V&&>::value, "Value must be nothrow constructible from V&&");

    auto state = InterruptibleState<Value>::Make();
    V source{42};

    EXPECT_TRUE(state->SetValue(std::move(source)));
    ASSERT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().GetValue(), 42);
}

TEST(InterruptibleStateSetValueTest, SetValueAcceptsPotentiallyThrowingValueConstructor)
{
    using Value = ThrowingValueConstructibleFromMoveSource;
    using V = MoveSource;

    static_assert(std::is_constructible<Value, V&&>::value, "Value must be constructible from V&&");
    static_assert(!std::is_nothrow_constructible<Value, V&&>::value,
                  "Value must be potentially throwing when constructed from V&&");

    auto state = InterruptibleState<Value>::Make();
    V source{84};

    EXPECT_TRUE(state->SetValue(std::move(source)));
    ASSERT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().GetValue(), 84);
}

// Test SetValue move with derived type when base is move-constructible
TEST(InterruptibleStateSetValueTest, SetValueDerivedMoveTwiceReturnsFalse)
{
    auto state = InterruptibleState<BaseType>::Make();
    DerivedType derived1{10};

    EXPECT_TRUE(state->SetValue(std::move(derived1)));
    EXPECT_EQ(state->GetValue().value().value, 10);

    // Second attempt with different derived
    DerivedType derived2{20};
    EXPECT_FALSE(state->SetValue(std::move(derived2)));

    // Original value unchanged
    EXPECT_EQ(state->GetValue().value().value, 10);
}

// Test SetValue copy then try move with different types
TEST(InterruptibleStateSetValueTest, SetValueMixedTypesAttempts)
{
    auto state = InterruptibleState<long>::Make();

    // First with short (copy)
    const short s = 10;
    EXPECT_TRUE(state->SetValue(s));
    EXPECT_EQ(state->GetValue().value(), 10L);

    // Try again with int (move) - fails
    int i = 20;
    EXPECT_FALSE(state->SetValue(std::move(i)));
    EXPECT_EQ(state->GetValue().value(), 10L);
}

// Test SetValue with rvalue derived type
TEST(InterruptibleStateSetValueTest, SetValueRvalueDerivedToBase)
{
    auto state = InterruptibleState<BaseType>::Make();

    // Create temporary derived and move into SetValue
    EXPECT_TRUE(state->SetValue(DerivedType{555}));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().value, 555);
}

// Test SetValue with reference specialization and base type
TEST(InterruptibleStateSetValueTest, SetValueReferenceBaseType)
{
    auto state = InterruptibleState<BaseType&>::Make();
    BaseType base{123};

    EXPECT_TRUE(state->SetValue(base));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().get().value, 123);
}

// Test SetValue with unsigned to signed conversion (move)
TEST(InterruptibleStateSetValueTest, SetValueUnsignedToSignedMove)
{
    auto state = InterruptibleState<int>::Make();
    unsigned short us = 42;

    EXPECT_TRUE(state->SetValue(std::move(us)));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value(), 42);
}

// Test SetValue with unsigned to signed conversion (copy)
TEST(InterruptibleStateSetValueTest, SetValueUnsignedToSignedCopy)
{
    auto state = InterruptibleState<int>::Make();
    const unsigned char uc = 123;

    EXPECT_TRUE(state->SetValue(uc));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value(), 123);
}

// Test SetValue preserves type after implicit conversion
TEST(InterruptibleStateSetValueTest, SetValueImplicitConversionPreservesType)
{
    auto state = InterruptibleState<long>::Make();
    short s = 42;

    state->SetValue(std::move(s));

    // Value is stored as long, not short
    long result = state->GetValue().value();
    EXPECT_EQ(result, 42L);
    EXPECT_EQ(sizeof(result), sizeof(long));
}

// Type that is copy-constructible but NOT move-constructible
// This tests that the copy overload is used even when rvalue is passed
class CopyOnly
{
  public:
    explicit CopyOnly(int val) : value_(val) {}

    // Copy constructor - available
    CopyOnly(const CopyOnly& other) : value_(other.value_) {}
    CopyOnly& operator=(const CopyOnly& other)
    {
        value_ = other.value_;
        return *this;
    }

    // Move operations explicitly deleted - NOT move-constructible
    CopyOnly(CopyOnly&&) = delete;
    CopyOnly& operator=(CopyOnly&&) = delete;

    int GetValue() const { return value_; }

  private:
    int value_;
};

// Test SetValue with copy-only type (lvalue reference)
TEST(InterruptibleStateSetValueTest, SetValueCopyOnlyLvalue)
{
    auto state = InterruptibleState<CopyOnly>::Make();
    CopyOnly value{42};

    // SetValue with lvalue - copy overload used
    EXPECT_TRUE(state->SetValue(value));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().GetValue(), 42);
}

// Test SetValue with copy-only type (const reference)
TEST(InterruptibleStateSetValueTest, SetValueCopyOnlyConstRef)
{
    auto state = InterruptibleState<CopyOnly>::Make();
    const CopyOnly value{99};

    // SetValue with const lvalue - copy overload used
    EXPECT_TRUE(state->SetValue(value));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().GetValue(), 99);
}

// Note: Idempotent behavior with copy-only types is covered by generic tests

// Type with deleted default constructor - must be copy constructible
class CopyOnlyNotDefaultConstructible
{
  public:
    explicit CopyOnlyNotDefaultConstructible(int val) : value_(val) {}

    // Explicitly delete default constructor
    CopyOnlyNotDefaultConstructible() = delete;

    // Allow copy
    CopyOnlyNotDefaultConstructible(const CopyOnlyNotDefaultConstructible& other) : value_(other.value_) {}
    CopyOnlyNotDefaultConstructible& operator=(const CopyOnlyNotDefaultConstructible& other)
    {
        value_ = other.value_;
        return *this;
    }

    // Delete move
    CopyOnlyNotDefaultConstructible(CopyOnlyNotDefaultConstructible&&) = delete;
    CopyOnlyNotDefaultConstructible& operator=(CopyOnlyNotDefaultConstructible&&) = delete;

    int GetValue() const { return value_; }

  private:
    int value_;
};

// Test SetValue with no-default type
TEST(InterruptibleStateSetValueTest, SetValueCopyOnlyNotDefaultConstructible)
{
    auto state = InterruptibleState<CopyOnlyNotDefaultConstructible>::Make();
    CopyOnlyNotDefaultConstructible value{123};

    // Only copy constructor available
    EXPECT_TRUE(state->SetValue(value));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().GetValue(), 123);
}

// Test SetValue copy-only with SetError interaction
TEST(InterruptibleStateSetValueTest, SetValueCopyOnlySetErrorInteraction)
{
    auto state = InterruptibleState<CopyOnly>::Make();
    CopyOnly value{42};

    // SetValue succeeds
    EXPECT_TRUE(state->SetValue(value));

    // SetError fails
    EXPECT_FALSE(state->SetError(Error::kUnset));
    EXPECT_EQ(state->GetValue().value().GetValue(), 42);
}

// Test SetValue with copy-only type in reference specialization
TEST(InterruptibleStateSetValueTest, SetValueCopyOnlyReference)
{
    auto state = InterruptibleState<CopyOnly&>::Make();
    CopyOnly value{777};

    // Reference specialization stores reference_wrapper to copy-only type
    EXPECT_TRUE(state->SetValue(value));
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().get().GetValue(), 777);
}

// Test SetValue copy-only preserves value over scope boundaries
TEST(InterruptibleStateSetValueTest, SetValueCopyOnlyPreservesOverScope)
{
    auto state = InterruptibleState<CopyOnly>::Make();

    {
        CopyOnly value{42};
        EXPECT_TRUE(state->SetValue(value));
    }  // value goes out of scope

    // Stored copy still accessible
    EXPECT_TRUE(state->GetValue().has_value());
    EXPECT_EQ(state->GetValue().value().GetValue(), 42);
}

}  // namespace
}  // namespace detail
}  // namespace concurrency
}  // namespace score
