?($	ܣ??"???L7??&??Qk?w????!;pΈ????$	C{?v @???+?
@???O~???!Y????)@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$B>?٬????St$????A]?Fx??Y?=yX???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&!?lV}??ڬ?\m???A?o_???Ye?X???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????????A???????Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Fx$??.?!??u??A??%䃞??Yvq?-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O??e???mV}??b??A???S???Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???B?i???):????A#J{?/L??YHP?sג?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?s????F??_???AK?46??Y=?U?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&;pΈ???????????AW[??????Y??_?L??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&k?w??#??=
ףp=??A?{??Pk??Yc?ZB>???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?0?*?????	h"??AK?=?U??Y䃞ͪϕ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
)?Ǻ???q=
ףp??A?1w-!??Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&؁sF??????ʡE??A^K?=???Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&|??Pk???_)?Ǻ??A?HP???YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???h o??)\???(??A???QI??YA??ǘ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&P??n?????ڊ?e??AC??6??YtF??_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&{?G?z??d?]K???AL?
F%u??YZd;?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??m4????Tt$?????A\ A?c???Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Ǻ???????3???A???????Y??ǘ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??<,Ԛ????|гY??A'???????Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&e?X???!?rh????AX9??v???Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Qk?w???????_vO??A?\m?????Y?q??????*	43333??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???S????!?=N?E:A@)?o_???151gu?G?@:Preprocessing2F
Iterator::Model???????!\?A?c?@@)??x?&1??1?u?-4@:Preprocessing2U
Iterator::Model::ParallelMapV20L?
F%??!?zZ*@)0L?
F%??1?zZ*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip46<???!R_3N?P@)?5?;Nѱ?1[? Z4&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??6???!???k?J!@)??6???1???k?J!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?? ?	??!'i??90@)R???Q??1??c.?N@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap]?C?????!Ė$?5@)?Zd;??1?kE??u@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*8??d?`??!?S??e	@)8??d?`??1?S??e	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???i??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?mY?c???b­x?????_vO??!???????	!       "	!       *	!       2$	?_\?O??[$??6\???\m?????!W[??????:	!       B	!       J$	n?????.D??y?y?&1???!?=yX???R	!       Z$	n?????.D??y?y?&1???!?=yX???JCPU_ONLYY???i??@b Y      Y@q??$?kV@"?
both?Your program is POTENTIALLY input-bound because 53.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?89.6722% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 