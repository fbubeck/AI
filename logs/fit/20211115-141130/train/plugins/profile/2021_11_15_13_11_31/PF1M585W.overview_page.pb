?($	w?,??4??B|.ރ???O??e??!??ZӼ???$	!?Oo4?@???5?@v*q4|@!??S?{.@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?O??e?????????A6?;Nё??Y:??H???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&H?z?G????@?????A?????B??YQ?|a??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&2U0*???2??%????A?-????Y+??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&jM??S???A?f???A????x???Y2??%䃞?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ZӼ???1?Zd??Aۊ?e????Y
ףp=
??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M?O?????_vO??A?1??%???Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????K7???>W[????A      ??Y	?c???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&0L?
F%????JY?8??A??V?/???Ym???{???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&}?5^?I??2??%????A46<???Y&S????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	.?!??u???$??C??Avq?-??Y??&???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
x$(~??"?uq??A??C?l??Y?+e?X??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????N@??A??^)??YˡE?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Q??????m4??@??A??????Yg??j+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ǘ?????&S???A?46<??Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+????	?c???A??e??a??Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&8??d?`????a??4??A??	h"??Y?e??a???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&鷯????? ?	??A?&?W??Y??ͪ?Ֆ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&y?&1???      ??A?U??????Yg??j+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????????(??A:#J{?/??Y?sF????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Y?8??m???o_???AX?5?;N??YD?l?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&7?[ A????m4????AW[??????YM?O???*	?????G?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??+e???!?'B?@@)1?Zd??1mR??D?@:Preprocessing2F
Iterator::ModelB>?٬???!VcH~?@@)?/?$??1?b:??,4@:Preprocessing2U
Iterator::Model::ParallelMapV2?a??4???!?Ǭ?vN+@)?a??4???1?Ǭ?vN+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??ܵ???!U?????P@)???z6??1?&?N??%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice5?8EGr??!?@3N?$@)5?8EGr??1?@3N?$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??D????!@?ԃ%?0@)8gDio??1?/?re?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapl	??g???!???C ?5@)
ףp=
??1(5?j?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*$????ۗ?!?/?<@)$????ۗ?1?/?<@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?'ɠP@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	O\'?????=kx???????????!??_vO??	!       "	!       *	!       2$	y-!?l????e,0ְ?6?;Nё??!ۊ?e????:	!       B	!       J$	????ؠ???/z?????ZӼ???!:??H???R	!       Z$	????ؠ???/z?????ZӼ???!:??H???JCPU_ONLYY?'ɠP@b Y      Y@qlݘC@"?
both?Your program is POTENTIALLY input-bound because 53.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?38.0594% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 