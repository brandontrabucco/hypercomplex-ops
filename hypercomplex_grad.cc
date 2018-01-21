#include "tensorflow/core/user_ops/hypercomplex_conjugate_op.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
    typedef FunctionDefHelper FDH;

    Status HypercomplexConjugateGrad(
        const AttrSlice& attrs,
        FunctionDef* g
    ) {
        *g = FDH::Define(
            "HypercomplexConjugateGrad",
            {"conjugate_grad: T"},
            {"to_conjugate_grad: T"},
            {{"T: {uint8, int8, int16, int32, float, double} = DT_FLOAT"}},
            {{
                {"to_conjugate_grad"},
                "HypercomplexConjugate",
                {"conjugate_grad"},
                {{"T", "$T"}}}});
        return Status::OK();
    }

    Status HypercomplexMultiplyGrad(
        const AttrSlice& attrs,
        FunctionDef* g
    ) {
        *g = FDH::Define(
            "HypercomplexMultiplyGrad",
            {"product_grad: T"},
            {"left_factor_grad: T", "right_factor_grad"},
            {{"T: {uint8, int8, int16, int32, float, double} = DT_FLOAT"}},
            {
                {{"left_factor_conj"}, "HypercomplexConjugate", {"left_factor_grad"}, {{"T", "$T"}}},
                {{"right_factor_conj"}, "HypercomplexConjugate", {"right_factor_grad"}, {{"T", "$T"}}},
                {{"left_factor_grad"}, "HypercomplexMultiply", {"left_factor_conj", "product_grad"}, {{"T", "$T"}}},
                {{"right_factor_grad"}, "HypercomplexMultiply", {"product_grad", "right_factor_conj"}, {{"T", "$T"}}}});
        return Status::OK();
    }

    REGISTER_OP_GRADIENT("HypercomplexConjugate", HypercomplexConjugateGrad);
    REGISTER_OP_GRADIENT("HypercomplexMultiply", HypercomplexMultiplyGrad);
}