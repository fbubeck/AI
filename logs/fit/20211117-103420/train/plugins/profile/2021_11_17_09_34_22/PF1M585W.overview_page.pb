?($	ۃ??y???O???????2ı.n??!??ׁsF
@$	t	?'?@S5d??@q???_??!???3r?0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$	?^)???h"lxz???ADio?????Y??N@a??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?3??7???<?R?!???A?'????Y?a??4???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???JY????Fx$??A`??"????YA??ǘ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?8??m4??????<,??A(~??k	??Yt$???~??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?=yX???
h"lxz??A??:M??Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&J+????-????A????<,??Y??ڊ?e??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?<,Ԛ???L7?A`???A{?/L?
??YGx$(??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ׁsF
@,Ԛ????A??z6?@Y??e?c]??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?\?C????}?5^?I??A?ͪ??V??Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	ڬ?\m???Gx$(??A?0?*??Y??d?`T??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
????????W?2???A?W?2??Ye?X???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?a??4????D???J??A??QI????YQ?|a2??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??:M??X9??v??A????9#??Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???(????=yX???AgDio????YǺ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ӽ?????? ?	??A?QI??&??Y?f??j+??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+?????$(~????Ad?]K???Yj?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????@:#J{??@A?;Nё\??Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Nё\?C??yX?5?;??A0?'???Y?g??s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X9??v???F????x??A?!?uq??Y??ڊ?e??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?N@a??????B?i??A?????K??YHP?s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?2ı.n??-!?lV??A?Pk?w???Ya2U0*???*	     T?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat1?Zd??!?m??LC@)???߾??1Z??ǎA@:Preprocessing2F
Iterator::Model?1??%???!dM?moA@)j?q?????1?~L?e?4@:Preprocessing2U
Iterator::Model::ParallelMapV2??H.???!?7???l,@)??H.???1?7???l,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?c?]K???!N?wIHP@)HP?s???1ٺ???$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?c]?F??!p?I?A@)?c]?F??1p?I?A@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?;Nё\??!?ܳ?^s(@)??ܵ??1W???,?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???Q???!ͫ????0@)??m4????1n???LG@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??|гY??!*>ѰQX@)??|гY??1*>ѰQX@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 57.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?o[?S@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??7???3&lh???-!?lV??!:#J{??@	!       "	!       *	!       2$	\u?&9??a?? ?????Pk?w???!??z6?@:	!       B	!       J$	?X?O????0?????e?X???!??N@a??R	!       Z$	?X?O????0?????e?X???!??N@a??JCPU_ONLYY?o[?S@b Y      Y@qr???A>@"?
both?Your program is POTENTIALLY input-bound because 57.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?30.2569% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 