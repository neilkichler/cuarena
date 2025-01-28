#ifndef CUARENA_DEFER_H
#define CUARENA_DEFER_H

#include <utility>

#define DEFER_1(x, y) x##y
#define DEFER_2(x, y) DEFER_1(x, y)
#define DEFER_3(x)    DEFER_2(x, __COUNTER__)
#define defer(code)   auto DEFER_3(_defer_) = finally([&]() { code; })

namespace internal
{
// final_action allows you to ensure something gets run at the end of a scope
template<class F>
class final_action {
public:
    static_assert(!std::is_reference_v<F> && !std::is_const_v<F> && !std::is_volatile_v<F>,
                  "Final_action should store its callable by value");

    explicit final_action(F f) noexcept
        : f_(std::move(f))
    { }

    final_action(final_action &&other) noexcept
        : f_(std::move(other.f_))
        , invoke_(std::exchange(other.invoke_, false))
    { }

    final_action(const final_action &)            = delete;
    final_action &operator=(const final_action &) = delete;
    final_action &operator=(final_action &&)      = delete;

    ~final_action() noexcept
    {
        if (invoke_)
            f_();
    }

private:
    F f_;
    bool invoke_ { true };
};
} // namespace internal

// finally() - convenience function to generate a final_action
template<class F>
[[nodiscard]] auto finally(F &&f) noexcept
{
    return internal::final_action<std::remove_cv_t<std::remove_reference_t<F>>>(
        std::forward<F>(f));
}

#endif // CUARENA_DEFER_H
