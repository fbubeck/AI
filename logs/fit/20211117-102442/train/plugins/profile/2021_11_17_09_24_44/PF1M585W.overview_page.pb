?($	?V???	???8????;?O??n??!p_?Q @$	??{?*+@??"d*@S\3?p.??!?;?me0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?V?/?'??r?鷯??AHP?s???Y??QI????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&@a??+????????A?ݓ??Z??Y	??g????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&e?`TR'??6?>W[???A???B?i??Y?I+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???Mb???MbX9??AH?z?G??Y?MbX9??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V-?????h"lxz???A?&1???YԚ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??0?*???W?2??AY?8??m??Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&_?L?J???{??Pk??A"??u????Y?g??s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&S??:???? ?rh??A&S????Y???~?:??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& ?~?:p???\m?????Aꕲq???Y???Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?7??d???+????AY?? ???YB`??"۩?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
o??ʡ??㥛? ???A?B?i?q??Y?=yX???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&[????<??`??"????A??9#J{??YT㥛? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&1?*?????v??/??A'1?Z??Y?St$????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????H???q??????As??A???YB>?٬???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&p_?Q @??ܵ??A3ı.n???Y8gDio??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&`vOj???e?c]???AT㥛? ??Yq=
ףp??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?9#J{???m???????A1?*????Y??@??Ǩ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&B`??"???=?U????A c?ZB>??Y?/?$??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&6?>W[????<,Ԛ???A?[ A???Y?3??7??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????Mb???=yX???A???Q???Yı.n???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&;?O??n???:pΈҾ?A???ׁs??Yı.n???*	53333Ř@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??#?????!ނ?e<D@)sh??|???1??LŞ?B@:Preprocessing2F
Iterator::Model???????!KRO?@@)?6?[ ??1&???4@:Preprocessing2U
Iterator::Model::ParallelMapV2?~?:p???!?n&s(@)?~?:p???1?n&s(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip$???~???!Z?V???P@)?#??????1⫝̸B?#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice???B?i??!P?W???@)???B?i??1P?W???@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??d?`T??!L;H?o?)@)????1I?8?+?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??ZӼ???!?i????0@)\ A?c̭?1?`?^@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??#?????!ނ?e<@)??#?????1ނ?e<@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 58.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???<@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	RS?oj??z?Cր???:pΈҾ?!??ܵ??	!       "	!       *	!       2$	?v???cƒ5??????ׁs??! c?ZB>??:	!       B	!       J$	???3???5?o|???B>?٬???!Ԛ?????R	!       Z$	???3???5?o|???B>?٬???!Ԛ?????JCPU_ONLYY???<@b Y      Y@qL"n?>=@"?
both?Your program is POTENTIALLY input-bound because 58.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?29.2451% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 