#ifndef __INTEGRATOR_TABLES_H__
#define __INTEGRATOR_TABLES_H__

#include <array>

// Given a function f() its integral \int_K f(x)dx can be approximated using a quadrature rule. The general
// scheme of a qudrature formula for the approximation of integral \int_K f(x)dx is given by a finite sum
// \sum_{i=1}^N [f(x_i) * w_i] where x_i and w_i are properly choosen quadrature nodes and weights.

// this file is just a collection of weights and nodes for quadrature formulae to be applied on a reference
// mesh element (N-dimensional unit simplex). Dimensions 1,2 and 3 considered for various element orders

// N dimension of the integration domain, M number of nodes of the formula
template <unsigned int N, unsigned int M> struct IntegratorTable;

// the higher the number of quadrature nodes, the higher the numerical precision, but the higher also the
// computational cost to obtain the approximation

// references: "Numerical Models for Differential Problems, Alfio Quarteroni. Second edition"
//             "The finite element method: Linear static and dynamic finite element analysis, Thomas J.R. Hughes"

// 1D linear elements (gaussian integration)
// reference element: simplex of vertices (0), (1)

// 2 point formula
template <> struct IntegratorTable<1, 2> {
  // position of nodes (in barycentric coordinates)
  static constexpr std::array<std::array<double, 1>, 2> nodes = {
    {{0.211324865405187},
     {0.788675134594812}}
  };
  // weights of the quadrature rule
  static constexpr std::array<double, 2> weights = {
    {0.500000000000000,
     0.500000000000000}
  };
};

// 2 point formula
template <> struct IntegratorTable<1, 3> {
  // position of nodes (in barycentric coordinates)
  static constexpr std::array<std::array<double, 1>, 3> nodes = {
    {{0.112701665379258},
     {0.500000000000000},
     {0.887298334620741}}
  };
  // weights of the quadrature rule
  static constexpr std::array<double, 3> weights = {
    {0.277777777777778,
     0.444444444444444,
     0.277777777777778}
  };
};


// 2D triangular elements
// reference element: simplex of vertices (0,0), (1,0), (0,1)

// 1 point formula, degree of precision 1
template <> struct IntegratorTable<2, 1> {
  // position of nodes (in barycentric coordinates)
  static constexpr std::array<std::array<double, 2>, 1> nodes = {
    {{0.333333333333333, 0.333333333333333}}
  };
  // weights of the quadrature rule
  static constexpr std::array<double, 1> weights = {
    {1.}
  };
};

// 3 point formula, degree of precision 3
template <> struct IntegratorTable<2, 3> {
  // position of nodes (in barycentric coordinates)
  static constexpr std::array<std::array<double, 2>, 3> nodes = {
    {{0.166666666666667, 0.166666666666667},
     {0.666666666666667, 0.166666666666667},
     {0.166666666666667, 0.666666666666667}}
  };
  // weights of the quadrature rule
  static constexpr std::array<double, 3> weights = {
    {0.333333333333333,
     0.333333333333333,
     0.333333333333333}
  };  
};

// 6 point formula, degree of precision 4
template <> struct IntegratorTable<2, 6> {
  // position of nodes (in barycentric coordinates)
  static constexpr std::array<std::array<double, 2>, 6> nodes = {
    {{0.445948490915965, 0.445948490915965},
     {0.445948490915965, 0.108103018168070},
     {0.108103018168070, 0.445948490915965},
     {0.091576213509771, 0.091576213509771},
     {0.091576213509771, 0.816847572980459},
     {0.816847572980459, 0.091576213509771}}
  };
  // weights of the quadrature rule
  static constexpr std::array<double, 6> weights = {
    {0.223381589678011,
     0.223381589678011,
     0.223381589678011,
     0.109951743655322,
     0.109951743655322,
     0.109951743655322}
  };  
};

// 7 point formula, degree of precision 5
template <> struct IntegratorTable<2, 7> {
  // position of nodes (in barycentric coordinates)
  static constexpr std::array<std::array<double, 2>, 7> nodes = {
    {{0.333333333333333,0.333333333333333},
     {0.101286507323456,0.101286507323456},
     {0.101286507323456,0.797426985353087},
     {0.797426985353087,0.101286507323456},
     {0.470142064105115,0.470142064105115},
     {0.470142064105115,0.059715871789770},
     {0.059715871789770,0.470142064105115}}
  };
  // weights of the quadrature rule
  static constexpr std::array<double, 7> weights = {
    {0.225000000000000,
     0.125939180544827,
     0.125939180544827,
     0.125939180544827,
     0.132394152788506,
     0.132394152788506,
     0.132394152788506}
  };  
};

// 12 point formula, degree of precision 6
template <> struct IntegratorTable<2, 12> {
  // position of nodes (in barycentric coordinates)
  static constexpr std::array<std::array<double, 2>, 12> nodes = {
    {{0.873821971016996, 0.063089017791802},
     {0.063089017791802, 0.873821971016996},
     {0.063089017791802, 0.063089017791802},
     {0.501426509658179, 0.249286745170910},
     {0.249286745170910, 0.501426509658179},
     {0.249286745170910, 0.249286745170910},
     {0.636502499121399, 0.310352451033785},
     {0.310352451033785, 0.636502499121399},
     {0.636502499121399, 0.053145049844816},
     {0.053145049844816, 0.636502499121399},
     {0.310352451033785, 0.053145049844816},
     {0.053145049844816, 0.310352451033785}}
  };
  // weights of the quadrature rule
  static constexpr std::array<double, 12> weights = {
    {0.050844906370207,
     0.050844906370207,
     0.050844906370207,
     0.116786275726397,
     0.116786275726397,
     0.116786275726397,
     0.082851075618374,
     0.082851075618374,
     0.082851075618374,
     0.082851075618374,
     0.082851075618374,
     0.082851075618374}
  };  
};

// 3D tetrahedric elements
// reference element: simplex of vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1)

// 1 point formula, degree of precision 1
template <> struct IntegratorTable<3, 1> {
  // position of nodes (in barycentric coordinates)
  static constexpr std::array<std::array<double, 3>, 1> nodes = {
    {{0.250000000000000, 0.250000000000000, 0.250000000000000}}
  };
  // weights of the quadrature rule
  static constexpr std::array<double, 1> weights = {
    {1.}
  };
};

// 4 point formula, degree of precision 3
template <> struct IntegratorTable<3, 4> {
  // position of nodes (in barycentric coordinates)
  static constexpr std::array<std::array<double, 3>, 4> nodes = {
    {{0.585410196624969,0.138196601125011,0.138196601125011},
     {0.138196601125011,0.138196601125011,0.138196601125011},
     {0.138196601125011,0.138196601125011,0.585410196624969},
     {0.138196601125011,0.585410196624969,0.138196601125011}}
  };
  // weights of the quadrature rule
  static constexpr std::array<double, 4> weights = {
    {0.250000000000000,
     0.250000000000000,
     0.250000000000000,
     0.250000000000000}
  };
};

// 5 point formula, degree of precision 4
template <> struct IntegratorTable<3, 5> {
  // position of nodes (in barycentric coordinates)
  static constexpr std::array<std::array<double, 3>, 5> nodes = {
    {{0.250000000000000, 0.250000000000000, 0.250000000000000},
     {0.500000000000000, 0.166666666666667, 0.166666666666667},
     {0.166666666666667, 0.500000000000000, 0.166666666666667},
     {0.166666666666667, 0.166666666666667, 0.500000000000000},
     {0.166666666666667, 0.166666666666667, 0.166666666666667}}
  };
  // weights of the quadrature rule
  static constexpr std::array<double, 5> weights = {
    {-0.80000000000000,
     0.450000000000000,
     0.450000000000000,
     0.450000000000000,
     0.450000000000000}
  };
};

#endif // __INTEGRATOR_TABLES_H__