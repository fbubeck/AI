$	:m????`WA'B??'1?Z??!z6?>W??$	::@???@?]Fʸ?@?I?????!Ⱦ??z+*@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Gx$(??=,Ԛ???A?D?????Yo?ŏ1??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?K7?A`??M?J???AQ?|a2??Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???B?i??X?2ı.??AΈ?????Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&z6?>W[??<?R?!???AC??6??YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??e??a??,e?X??A?-???1??Y????Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???ZӼ??Tt$?????A+????Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???(????W?2ı??A???z6??Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&#??~j???????<,??A?8??m4??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?? ?	??&䃞ͪ??A?D?????Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?Q????????QI???AV}??b??Y???S㥛?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
|a2U0????ͪ????A)\???(??Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&d;?O??????K7?A??A?U??????Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&z6?>W??0*??D??A??????Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&>?٬?\??
h"lxz??A?鷯??Y_?Qڛ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?v??/???/L?
F??A?Zd;??Y??&???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&J{?/L???jM??S??Au????Y9??v????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?x?&1??46<?R??A??	h"??Yݵ?|г??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&L7?A`????C??????A??<,Ԛ??YǺ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????ݵ?|г??A?????B??Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&B?f??j??Gx$(??Af?c]?F??Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&'1?Z???HP???A????????Y?l??????*	fffff??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?]K?=??!=?^???@@)?o_???1?6?	#?@:Preprocessing2F
Iterator::Model/?$???!d܍k?LB@)????????1F??N"6@:Preprocessing2U
Iterator::Model::ParallelMapV2?d?`TR??!?	??,@)?d?`TR??1?	??,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip<?R?!???!?#r?V?O@)io???T??1???(?%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?I+???!/??c?@)?I+???1/??c?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate????!??????+@)I.?!????1ǡ"?*?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???B?i??!?4?)??2@)?|a2U??1?q-??B@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?o_???!1?7??6@)?o_???11?7??6@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 50.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9\?J???@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	eu?WH'???Z/????=,Ԛ???!0*??D??	!       "	!       *	!       2$	^{O?????h?d??????????!f?c]?F??:	!       B	!       J$	?4{;??=?v?]?{?ŏ1w-!??!o?ŏ1??R	!       Z$	?4{;??=?v?]?{?ŏ1w-!??!o?ŏ1??JCPU_ONLYY\?J???@b 