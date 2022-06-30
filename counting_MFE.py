import loopy as lp
import numpy as np
from pyop2.codegen.loopycompat import _match_caller_callee_argument_dimension_
lp.Options.enforce_variable_access_ordered=False

slate_wrapper_knl = lp.make_kernel(
  ["[end, start] -> { [n] : start <= n < end }",
    "{ [i0] : i0 = 0 }",
    "{ [i8] : 0 <= i8 <= 9 }",
    "{ [i4] : i4 = 0 }",
    "{ [i7] : i7 = 0 }",
    "{ [i1] : 0 <= i1 <= 2 }",
    "{ [i6] : 0 <= i6 <= 1 }",
    "{ [i3] : i3 = 0 }",
    "{ [i2] : 0 <= i2 <= 9 }",
    "{ [i5] : i5 = 0 }",
    "{ [i9] : 0 <= i9 <= 9 }",
    "{ [i10] : i10 = 0 }",
    "{ [i11] : i11 = 0 }",
    "{ [i12] : i12 = 0 }",
    "{ [i13] : 0 <= i13 <= 9 }",
    "{ [i14] : i14 = 0 }",
    "{ [i15] : i15 = 0 }",
    "{ [i16] : 0 <= i16 <= 2 }",
    "{ [i17] : 0 <= i17 <= 1 }",
    "{ [i18] : i18 = 0 }",
    "{ [i19] : 0 <= i19 <= 9 }",
    "{ [i20] : i20 = 0 }"
  ],
  """ 
  for n
    for i20
      for i19
        for i18
          <> t2[i18, i19, i20] = dat2[map0[n, i19], i20]  {id=statement0}
        end
      end
    end
    for i15
      for i17
        for i16
          <> t1[i15, i16, i17] = dat1[map1[n, i16], i17]  {id=statement1}
        end
      end
    end
    for i14
      for i13
        for i12
          <> t0[i12, i13, i14] = 0  {id=statement2}
        end
      end
    end
   [i0,i8,i4]: t0[i0, i8, i4] = slate_loopy_knl_0([i0,i8,i4]: t0[i0, i8, i4] , [i7,i1,i6]: t1[i7, i1, i6], [i3,i2,i5]: t2[i3, i2, i5])  {id=statement4}
    for i11
      for i10
        for i9
          dat0[map0[n, i9], i10] = dat0[map0[n, i9], i10] + t0[i11, i9, i10]  {id=statement3, dep=statement2}
        end
      end
    end
  end
  """,
  [
  lp.ValueArg("start,end", "int32"),
  lp.GlobalArg("dat0", shape=(100, 1), dtype="float64"),
  lp.GlobalArg("dat1", shape=(100, 2), dtype="float64"),
  lp.GlobalArg("dat2", shape=(100, 1), dtype="float64"),
  lp.GlobalArg("map0", dtype="int32", shape=(100, 10)),
  lp.GlobalArg("map1", dtype="int32", shape=(100, 3)),
  ],
  assumptions="start>=0 and end<10000",
  target=lp.CTarget(),
  lang_version=(2018, 2),
  name="wrap_slate_loopy_knl_0",
  seq_dependencies=True)

print("Generated slate wrapper knl:\n", slate_wrapper_knl) 

slate_knl = lp.make_function(
  [
  "{ [i] : 0 <= i <= 9 }",
  "{ [i_3] : 0 <= i_3 <= 9 }",
  "{ [id_2] : 0 <= id_2 <= 9 }",
  "{ [id_3] : 0 <= id_3 <= 9 }",
  ],
  """
  for id_3
    <> T16[0 + id_3] = w_0[id_3 // 10, id_3 + (-10)*(id_3 // 10), 0]  {id=init_T16_0}
  end
  [i]: S0[i] = some_insns_knl([id_2]: T16[id_2])  {id=insn}
  for i_3
    output[i_3 // 10, i_3 + (-10)*(i_3 // 10), 0] = output[i_3 // 10, i_3 + (-10)*(i_3 // 10), 0] + S0[i_3]  {id=insn_0}
  end
  """,
  [
  lp.GlobalArg("output", shape=(1, 10, 1), dtype="float64"),
  lp.GlobalArg("coords", shape=(1, 3, 2), dtype="float64"),
  lp.GlobalArg("w_0", shape=(1, 10, 1), dtype="float64"),
  lp.TemporaryVariable("S0", dtype="float64", address_space=lp.AddressSpace.LOCAL),
  ],
  target=lp.CTarget(),
  lang_version=(2018, 2),
  name="slate_loopy_knl_0",
  seq_dependencies=True)

print("Generated slate knl:\n", slate_knl) 

knl = lp.make_function(
  """
  {[i_0, i_11, i_17]:
    0<=i_0,i_11, i_17<n}
  """,
  [
  f"""
  for i_0
  x[i_0] = -b[i_0] + b[i_0] {{id=x0}}
  end
  for i_11
  output[i_11] = x[i_11] {{id=out}}
  end
  """],   
  [
  lp.GlobalArg("output", shape=(10), dtype="float64"),
  lp.GlobalArg("b", shape=(10), dtype="float64"),
  lp.TemporaryVariable('x', dtype="float64",  address_space=lp.AddressSpace.LOCAL),
  ],
  target=lp.CTarget(),
  name="some_insns_knl",
  lang_version=(2018, 2),
  silenced_warnings=["single_writer_after_creation", "unused_inames"],
  seq_dependencies=True)

local_matfree_knl = lp.fix_parameters(knl, n=10)
print("Generated solve knl:\n", local_matfree_knl) 

slate_wrapper_knl =  lp.merge([slate_wrapper_knl, slate_knl])
slate_wrapper_knl = _match_caller_callee_argument_dimension_(slate_wrapper_knl, "slate_loopy_knl_0")
slate_wrapper_knl =  lp.merge([slate_wrapper_knl, local_matfree_knl])
slate_wrapper_knl = _match_caller_callee_argument_dimension_(slate_wrapper_knl, "some_insns_knl")
print("Merged kernel:\n", slate_wrapper_knl)

# inlining
def transform0(wrapper, knls):
  # inline all inner kernels
  for knl in knls:
    names = knl.callables_table
    for name in names:
        print(name)
        if (name in wrapper.callables_table.keys()
            and isinstance(wrapper.callables_table[name],
                            lp.CallableKernel)):
            wrapper = lp.inline_callable_kernel(wrapper, name)
  return wrapper

# vectorisation
def transform1(tunit):
    # this is a verbatim copy of the transformations in pyop2.global_kernel
    from functools import reduce
    import pymbolic.primitives as prim

    iname = "n"
    batch_size = 2
    alignment = 64

    if True:
        # loopy warns for every instruction that cannot be vectorized;
        # ignore in non-debug mode.
        new_entrypoint = tunit.default_entrypoint.copy(
            silenced_warnings=(tunit.default_entrypoint.silenced_warnings
                               + ["vectorize_failed"]))
        tunit = tunit.with_kernel(new_entrypoint)

    kernel = tunit.default_entrypoint

    # align temps
    tmps = {name: tv.copy(alignment=alignment)
            for name, tv in kernel.temporary_variables.items()}
    kernel = kernel.copy(temporary_variables=tmps)

    # {{{ record temps that cannot be vectorized

    # Do not vectorize temporaries used outside *iname*
    temps_not_to_vectorize = reduce(set.union,
                                    [(insn.dependency_names()
                                      & frozenset(kernel.temporary_variables))
                                     for insn in kernel.instructions
                                     if iname not in insn.within_inames],
                                    set())

    # Constant literal temporaries are arguments => cannot vectorize
    temps_not_to_vectorize |= {name
                               for name, tv in kernel.temporary_variables.items()
                               if (tv.read_only
                                   and tv.initializer is not None)}

    # {{{ clang (unlike gcc) does not allow taking address of vector-type
    # variable

    # FIXME: Perform this only if we know we are not using gcc.
    for insn in kernel.instructions:
        if (
                isinstance(insn, lp.MultiAssignmentBase)
                and isinstance(insn.expression, prim.Call)
                and insn.expression.function.name in ["solve", "inverse"]):
            temps_not_to_vectorize |= (insn.dependency_names())

    # }}}

    # }}}

    # {{{ TODO: placeholder until loopy's simplify_using_pwaff gets smarter

    # transform to ensure that the loop undergoing array expansion has a
    # lower bound of '0'
    from loopy.symbolic import pw_aff_to_expr
    lbound = pw_aff_to_expr(kernel.get_iname_bounds(iname).lower_bound_pw_aff)
    shifted_iname = kernel.get_var_name_generator()(f"{iname}_shift")
    kernel = lp.affine_map_inames(kernel, iname, shifted_iname,
                                  [(prim.Variable(shifted_iname),
                                   (prim.Variable(iname) - lbound))])

    # }}}

    # split iname
    # note there is no front slab needed because iname is shifted (see above)
    slabs = (0, 1)
    inner_iname = kernel.get_var_name_generator()(f"{shifted_iname}_batch")

    kernel = lp.split_iname(kernel, shifted_iname, batch_size, slabs=slabs,
                            inner_iname=inner_iname)

    # adds a new axis to the temporary and indexes it with the provided iname
    # i.e. stores the value at each instance of the loop. (i.e. array
    # expansion)
    kernel = lp.privatize_temporaries_with_inames(kernel, inner_iname)

    # tag axes of the temporaries as vectorised
    for name, tmp in kernel.temporary_variables.items():
        if name not in temps_not_to_vectorize:
            tag = (len(tmp.shape)-1)*"c," + "vec"
            kernel = lp.tag_array_axes(kernel, name, tag)

    # tag the inner iname as vectorized
    kernel = lp.tag_inames(kernel,
                           {inner_iname: lp.VectorizeTag(lp.OpenMPSIMDTag())})

    return tunit.with_kernel(kernel)

knl = slate_wrapper_knl.default_entrypoint
warnings = list(knl.silenced_warnings)
warnings.extend(["insn_count_subgroups_upper_bound", "no_lid_found"])
knl = knl.copy(silenced_warnings=warnings)
slate_wrapper_knl = slate_wrapper_knl.with_kernel(knl)
op_map = lp.get_op_map(slate_wrapper_knl, subgroup_size=1)
mem_map = lp.get_mem_access_map(slate_wrapper_knl, subgroup_size=1)  # Is subgroup_size=1 correct?

# inlined_knl = transform0(slate_wrapper_knl, [slate_knl, local_matfree_knl])
# print("Inlined knl:\n", inlined_knl)

print("MEMS= {0}".format(mem_map.filter_by(mtype=['global'], dtype=[np.float64]).eval_and_sum({})))
for op in ['sub', 'sub', 'mul', 'div']:
    print(op)
    print("{0}S= {1}".format(op.upper(), op_map.filter_by(name=[op]).eval_and_sum({})))
