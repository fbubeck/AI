?($	????N????8j??????~?:??!??ʡE@$	??ɲ?@?o???@?Ǐ?~??!;?o!r,@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?>W[?????A?f????AS??:??Y)\???(??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??MbX??????<,??ApΈ?????Y??ʡE???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Zd;????,C????AGr?????Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??m4????4??@????A??ڊ?e??Yr??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????&䃞ͪ??A?e??a???Y??W?2ġ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&#J{?/L???H?}8??A?rh??|??Yh??|?5??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&)??0????!?uq??A??h o???Y?ܵ?|У?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?? ?	???ׁsF???A????_v??Y???~?:??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&o???T???^K?=???A?? ?	??Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??m4????e?X???Aۊ?e????Y?v??/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
333333??؁sF????Ad;?O????Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??bٽ@???H??A??C?l??YO??e?c??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ı.n??????????A??B?i???Y0L?
F%??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??"??~???c?]K???A=,Ԛ???Y?{??Pk??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&>yX?5????uq???A??	h"??YV}??b??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ʡE@??MbX??A??(???YI.?!????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ǻ??????C?l???A??MbX??Y|??Pk???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O??e?c??
ףp=
??A9??v????Y??ǘ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?q?????? ?o_???A??7??d??Y???_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ZӼ????Fx$??A?~j?t???Y^K?=???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???~?:??s??A϶?A^?I+??YZd;?O???*	gffff??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?????M??!?ט=?B@)??	h"l??1KH??@@:Preprocessing2F
Iterator::Model^K?=???!??sQ??A@)?:pΈ???1csK?,95@:Preprocessing2U
Iterator::Model::ParallelMapV2?[ A?c??!?s8Փ,@)?[ A?c??1?s8Փ,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip/?$???!?FW"/P@)?A`??"??1SP,\?"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceY?8??m??!Io.?"@)Y?8??m??1Io.?"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateΈ?????!???:*@)D?l?????1????dS@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapףp=
???!w{??t+3@)6<?R???1??:Q8@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*%u???!???`Һ@)%u???1???`Һ@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?Ԇ%??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	s?F?R???M?1?Vt??s??A϶?!??MbX??	!       "	!       *	!       2$	?????P??¹??Cz??S??:??!??C?l??:	!       B	!       J$	???????ׯ?`l???v??/??!O??e?c??R	!       Z$	???????ׯ?`l???v??/??!O??e?c??JCPU_ONLYY?Ԇ%??@b Y      Y@q??!??6U@"?
both?Your program is POTENTIALLY input-bound because 55.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?84.856% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 